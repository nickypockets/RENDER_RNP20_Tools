import os
import json
from pathlib import Path
from typing import Optional, Literal, List

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Optional dependencies for advanced decomposition methods
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Load the canonical burns JSON (assumed present and structured as in data/burns_data/burns.json)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = PROJECT_ROOT / "data" / "burns_data" / "burns.json"
if not DATA_JSON.exists():
    raise FileNotFoundError(f"Expected burns JSON not found: {DATA_JSON}")

with open(DATA_JSON, "r", encoding="utf-8") as fh:
    obj = json.load(fh)

# the JSON uses a top-level "data" array of records
records = obj.get("data")
if not isinstance(records, list):
    raise ValueError(f"Unexpected JSON structure in {DATA_JSON}; expected 'data' list")

# Build a DataFrame from the records and normalise fields
df_burns = pd.DataFrame(records)

# prefer the explicit fields from the API
if "startDate" in df_burns.columns:
    df_burns["startDate"] = pd.to_datetime(df_burns["startDate"], utc=True, errors="coerce")
if "endDate" in df_burns.columns:
    df_burns["endDate"] = pd.to_datetime(df_burns["endDate"], utc=True, errors="coerce")

# choose numeric burn fields
if "burnedRender" in df_burns.columns:
    df_burns["Burns"] = pd.to_numeric(df_burns["burnedRender"], errors="coerce")
elif "totalRenderUsed" in df_burns.columns:
    df_burns["Burns"] = pd.to_numeric(df_burns["totalRenderUsed"], errors="coerce")
else:
    raise ValueError("No suitable burn numeric field found in burns JSON")

if "burnedRenderUSDCAmt" in df_burns.columns:
    df_burns["USDBurn"] = pd.to_numeric(df_burns["burnedRenderUSDCAmt"], errors="coerce")

# sort by startDate if present
if "startDate" in df_burns.columns:
    df_burns = df_burns.sort_values("startDate").reset_index(drop=True)


def compute_sma_and_quarterly_growth(df: pd.DataFrame, sma_window: int = 12, quarterly_periods: int = 3) -> pd.DataFrame:
    """Compute SMA over `sma_window` and rolling quarterly growth.

    - Adds a column `SMA_{sma_window}` containing the rolling mean of `Burns`.
    - Adds a column `Rolling_Quarterly_Growth` which is a % change vs `quarterly_periods` periods.

    The function mutates the provided DataFrame and also returns it for convenience.
    """
    if "Burns" not in df.columns:
        raise ValueError("DataFrame must contain a 'Burns' column to compute SMA and growth")

    sma_col = f"SMA_{sma_window}"
    # Compute moving average (SMA)
    df[sma_col] = df["Burns"].rolling(window=sma_window).mean()

    # Compute quarterly growth as percent change: current SMA vs SMA from `quarterly_periods` ago
    # This compares non-overlapping windows (e.g., current quarter vs previous quarter)
    prev_sma = df[sma_col].shift(quarterly_periods)
    df["Rolling_Quarterly_Growth"] = ((df[sma_col] / prev_sma) - 1) * 100

    return df


def compute_trend_seasonality_bands(
    df: pd.DataFrame,
    method: Literal["statsmodels", "moving_average", "savgol"] = "statsmodels",
    period: int = 12,
    sigma_levels: List[int] = [1, 2, 3]
) -> pd.DataFrame:
    """Compute trend + seasonality and standard deviation bands.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Burns' column
    method : str
        Decomposition method:
        - "statsmodels": Uses seasonal_decompose (additive)
        - "moving_average": Simple trend (moving average) + seasonal residuals
        - "savgol": Savitzky-Golay filter for trend + seasonal residuals
    period : int
        Seasonal period (default 12 for quarterly if data is monthly)
    sigma_levels : List[int]
        Which sigma levels to compute (e.g., [1, 2, 3])
    
    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - 'Trend': extracted trend
        - 'Seasonal': extracted seasonality
        - 'Expected': trend + seasonal
        - 'Residual': actual - expected
        - 'Sigma_1', 'Sigma_2', 'Sigma_3': standard deviation bands (if requested)
    """
    df = df.copy()
    burns = df["Burns"].fillna(method="ffill").fillna(method="bfill")
    
    if method == "statsmodels":
        if not HAS_STATSMODELS:
            print("Warning: statsmodels not installed. Install with: pip install statsmodels")
            print("Falling back to moving_average method")
            method = "moving_average"
        else:
            try:
                # Use statsmodels seasonal_decompose
                result = seasonal_decompose(burns, model="additive", period=period, extrapolate_trend="freq")
                df["Trend"] = result.trend
                df["Seasonal"] = result.seasonal
                df["Residual"] = result.resid
            except Exception as e:
                print(f"Warning: statsmodels decomposition failed ({e}), falling back to moving_average")
                method = "moving_average"
    
    if method == "moving_average":
        # Simple centered moving average for trend
        df["Trend"] = burns.rolling(window=period, center=True, min_periods=1).mean()
        detrended = burns - df["Trend"]
        # Extract seasonality by averaging each position in the cycle
        seasonal_pattern = np.array([detrended.iloc[i::period].mean() for i in range(period)])
        # Tile the pattern to match dataframe length
        df["Seasonal"] = np.tile(seasonal_pattern, len(df) // period + 1)[:len(df)]
        df["Residual"] = burns - df["Trend"] - df["Seasonal"]
    
    elif method == "savgol":
        if not HAS_SCIPY:
            print("Warning: scipy not installed. Install with: pip install scipy")
            print("Falling back to moving_average method")
            method = "moving_average"
        else:
            try:
                # Savitzky-Golay filter for smooth trend
                window = min(period * 2 + 1, len(burns) // 2 * 2 + 1)  # must be odd
                if window < 5:
                    window = 5
                df["Trend"] = signal.savgol_filter(burns, window_length=window, polyorder=2)
                detrended = burns - df["Trend"]
                # Extract seasonality
                seasonal_pattern = np.array([detrended.iloc[i::period].mean() for i in range(period)])
                df["Seasonal"] = np.tile(seasonal_pattern, len(df) // period + 1)[:len(df)]
                df["Residual"] = burns - df["Trend"] - df["Seasonal"]
            except Exception as e:
                print(f"Warning: savgol filter failed ({e}), falling back to moving_average")
                # Fallback to moving average
                method = "moving_average"
    
    if method == "moving_average":
        # Final fallback: simple centered moving average for trend
        df["Trend"] = burns.rolling(window=period, center=True, min_periods=1).mean()
        detrended = burns - df["Trend"]
        seasonal_pattern = np.array([detrended.iloc[i::period].mean() for i in range(period)])
        df["Seasonal"] = np.tile(seasonal_pattern, len(df) // period + 1)[:len(df)]
        df["Residual"] = burns - df["Trend"] - df["Seasonal"]
    
    # Expected value is trend + seasonality
    df["Expected"] = df["Trend"] + df["Seasonal"]
    
    # Calculate standard deviation of residuals
    residual_std = df["Residual"].std()
    
    # Add sigma bands around expected value
    for sigma in sigma_levels:
        df[f"Upper_Sigma_{sigma}"] = df["Expected"] + sigma * residual_std
        df[f"Lower_Sigma_{sigma}"] = df["Expected"] - sigma * residual_std
    
    return df


def create_burns_chart(
    df: pd.DataFrame,
    show_sigma_bands: bool = False,
    sigma_levels: List[int] = [1, 2, 3],
    decomposition_method: Literal["statsmodels", "moving_average", "savgol"] = "statsmodels",
    use_dates: bool = True,
    show_original_burns: bool = False
) -> go.Figure:
    """Create and return a Plotly Figure for Burns, SMA and Rolling Quarterly Growth.

    The function expects the DataFrame to contain:
    - 'Burns'
    - 'SMA_12' (or an SMA column named 'SMA_{N}')
    - 'Rolling_Quarterly_Growth'
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    show_sigma_bands : bool
        If True, adds standard deviation bands based on trend+seasonality
    sigma_levels : List[int]
        Which sigma levels to display (e.g., [1, 2, 3])
    decomposition_method : str
        Method for trend/seasonality extraction:
        - "statsmodels": seasonal_decompose (most accurate, requires statsmodels)
        - "moving_average": simple moving average trend (fastest, most robust)
        - "savgol": Savitzky-Golay filter (smooth trend, requires scipy)
    use_dates : bool
        If True, use month/year dates on X-axis (default). If False, use epoch numbers.
    show_original_burns : bool
        If True, plot the original burns line with markers (default False).
    """
    # attempt to find the SMA column; prefer SMA_12 if present
    sma_cols = [c for c in df.columns if c.startswith("SMA_")]
    sma_col = "SMA_12" if "SMA_12" in df.columns else (sma_cols[0] if sma_cols else None)
    
    # Determine x-axis data: use dates if available and use_dates=True, otherwise use epoch or index
    if use_dates and "startDate" in df.columns:
        x_axis = df["startDate"]
        x_title = "Date"
    elif "id" in df.columns:
        x_axis = df["id"]
        x_title = "Epoch"
    else:
        x_axis = df.index
        x_title = "Epoch"
    
    # Compute sigma bands if requested
    if show_sigma_bands:
        df = compute_trend_seasonality_bands(df, method=decomposition_method, period=12, sigma_levels=sigma_levels)

    fig = go.Figure()

    # Add sigma bands first (so they appear behind the main traces)
    if show_sigma_bands and "Expected" in df.columns:
        # Define colors for sigma levels (fading opacity)
        sigma_colors = {
            1: "rgba(255, 165, 0, 0.15)",   # Orange, 15% opacity
            2: "rgba(255, 165, 0, 0.10)",   # Orange, 10% opacity
            3: "rgba(255, 165, 0, 0.05)",   # Orange, 5% opacity
        }
        
        # Plot bands from outermost to innermost
        for sigma in sorted(sigma_levels, reverse=True):
            upper_col = f"Upper_Sigma_{sigma}"
            lower_col = f"Lower_Sigma_{sigma}"
            
            if upper_col in df.columns and lower_col in df.columns:
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=df[upper_col],
                    mode="lines",
                    name=f"+{sigma}σ",
                    line=dict(color="rgba(255, 165, 0, 0.3)", width=1, dash="dot"),
                    showlegend=True
                ))
                
                # Lower bound with fill
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=df[lower_col],
                    mode="lines",
                    name=f"-{sigma}σ",
                    line=dict(color="rgba(255, 165, 0, 0.3)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor=sigma_colors.get(sigma, "rgba(255, 165, 0, 0.05)"),
                    showlegend=True
                ))
        
        # Add the expected (trend + seasonality) line
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df["Expected"],
            mode="lines",
            name="Expected (Trend+Seasonal)",
            line=dict(color="#FFA500", width=2, dash="dash")
        ))
    
    # Original burns (optional)
    if show_original_burns:
        fig.add_trace(go.Scatter(x=x_axis, y=df["Burns"], mode="lines+markers", name="Original Burns", line=dict(color="#A5C8FF"), marker=dict(color="#A5C8FF")))

    # Quarterly SMA
    if sma_col is not None:
        fig.add_trace(go.Scatter(x=x_axis, y=df[sma_col], mode="lines", name="Quarterly Burns", line=dict(color="#FFB3C6")))

    # Rolling Quarterly Growth trace on secondary axis
    fig.add_trace(go.Scatter(x=x_axis, y=df.get("Rolling_Quarterly_Growth"), mode="lines", name="Quarterly Growth Rate", yaxis="y2", line=dict(color="#FDE68A")))

    # Layout
    fig.update_layout(
        title="Render Network Quarterly Burns + Growth Rate",
        xaxis_title=x_title,
        yaxis=dict(
            title="Burns",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            gridwidth=1,
            griddash="dash",
        ),
        yaxis2=dict(
            title="Rolling % Growth",
            overlaying="y",
            side="right",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            gridwidth=1,
            griddash="dot",
        ),
        legend_title="Legend",
        template=None,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=600,
        width=1000,
        legend=dict(x=1.05, y=1, xanchor="left"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.25)", gridwidth=1, griddash="dash", zeroline=False)

    return fig


if __name__ == "__main__":
    # Compute derived fields and render the chart when run as a script (not on import)
    df_burns = compute_sma_and_quarterly_growth(df_burns, sma_window=12, quarterly_periods=12)
    
    # ========== CONFIGURATION OPTIONS ==========
    # Set show_sigma_bands=True to enable standard deviation bands
    # Choose sigma_levels: which bands to show (e.g., [1, 2, 3] for all three)
    # Choose decomposition_method:
    #   - "statsmodels": Most accurate, uses seasonal_decompose (requires statsmodels package)
    #   - "moving_average": Simplest, most robust, uses rolling average
    #   - "savgol": Smooth trend using Savitzky-Golay filter (requires scipy package)
    # Set use_dates=True to show month/year dates on X-axis, or False for epoch numbers
    # Set show_original_burns=True to plot the raw burns line with markers
    
    SHOW_SIGMA_BANDS = False
    SIGMA_LEVELS = [1, 2, 3]  # Options: [1], [1, 2], [1, 2, 3], [2, 3], etc.
    DECOMPOSITION_METHOD = "statsmodels"  # Options: "statsmodels", "moving_average", "savgol"
    USE_DATES = False  # True = show dates, False = show epoch numbers
    SHOW_ORIGINAL_BURNS = False  # True = show original burns line, False = hide it
    
    # ==========================================
    
    fig = create_burns_chart(
        df_burns,
        show_sigma_bands=SHOW_SIGMA_BANDS,
        sigma_levels=SIGMA_LEVELS,
        decomposition_method=DECOMPOSITION_METHOD,
        use_dates=USE_DATES,
        show_original_burns=SHOW_ORIGINAL_BURNS
    )
    
    # Save the chart to reports folder
    from pathlib import Path
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "burns_chart.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved burns chart to: {output_path}")
    
    fig.show()

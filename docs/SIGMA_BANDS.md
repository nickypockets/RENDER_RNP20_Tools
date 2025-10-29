# Sigma Bands for Burns Analysis

Statistical deviation bands help identify when actual burns deviate significantly from expected patterns.

## Quick Start

Edit `tools/burnProcessing.py` configuration:

```python
SHOW_SIGMA_BANDS = True
SIGMA_LEVELS = [1, 2, 3]
DECOMPOSITION_METHOD = "moving_average"
```

Then run:
```bash
python -m tools.burnProcessing
```

## Configuration

```python
from tools.burnProcessing import create_burns_chart

fig = create_burns_chart(
    df=df_burns,
    show_sigma_bands=True,              # Enable bands
    sigma_levels=[1, 2, 3],             # Which bands (1σ, 2σ, 3σ)
    decomposition_method='moving_average'  # Trend extraction method
)
```

### Decomposition Methods

| Method | Best For | Dependencies |
|--------|----------|--------------|
| **`moving_average`** | Production use (fast, robust) | None ✅ |
| `statsmodels` | Maximum accuracy | `pip install statsmodels` |
| `savgol` | Smooth trends | `pip install scipy` |

### Sigma Levels

- **1σ (68%)**: Normal variation - inner band
- **2σ (95%)**: Significant deviation - middle band
- **3σ (99.7%)**: Extreme outlier - outer band

Common configurations:
```python
sigma_levels=[1]          # Conservative - only show normal range
sigma_levels=[1, 2, 3]    # Standard - show all bands (recommended)
sigma_levels=[2, 3]       # Focus on significant deviations only
```

## Interpretation

**Reading the bands:**

- **Within 1σ**: Normal variation (68% of historical data)
- **Between 1σ-2σ**: Moderate deviation (27% of data)
- **Between 2σ-3σ**: Significant deviation (4% of data)  
- **Beyond 3σ**: Extreme outlier (0.3% of data) - investigate!

**What to look for:**

- Burns consistently **above 2σ** → Sustained increase beyond seasonal patterns
- Burns **below 2σ** → Concerning decrease that warrants attention
- **Crossing between bands** → Potential trend change

## Technical Details

### How It Works

1. **Extract Trend**: Uses rolling average (or statsmodels/savgol)
2. **Extract Seasonality**: Averages each position in seasonal cycle
3. **Calculate Expected**: Trend + Seasonality
4. **Measure Residuals**: Actual - Expected
5. **Compute σ**: Standard deviation of residuals
6. **Draw Bands**: Expected ± nσ for each level

### Visual Design

- Orange bands with fading opacity (3σ lightest, 1σ darkest)
- Dashed orange line shows expected value (trend + seasonality)
- Bands appear behind actual burns trace for clarity

### Data Requirements

- Minimum ~24 epochs recommended for seasonal patterns
- Works with missing data (forward/backward fill)
- Handles irregular time series automatically

## Examples

### Standard Configuration
```python
# All bands with simple method (no dependencies)
fig = create_burns_chart(
    df, 
    show_sigma_bands=True, 
    sigma_levels=[1, 2, 3],
    decomposition_method="moving_average"
)
```

### High-Precision Analysis
```python
# Using statsmodels for maximum accuracy
fig = create_burns_chart(
    df,
    show_sigma_bands=True,
    sigma_levels=[1, 2, 3],
    decomposition_method="statsmodels"
)
```

### Focus on Outliers
```python
# Only show significant deviations
fig = create_burns_chart(
    df,
    show_sigma_bands=True,
    sigma_levels=[2, 3],
    decomposition_method="moving_average"
)
```

## Troubleshooting

**Q: Bands look weird at the edges?**  
A: Normal - decomposition methods need a data window. First/last few points may be less accurate.

**Q: "statsmodels" or "scipy" not found?**  
A: Either install with `pip install statsmodels scipy` or use `decomposition_method="moving_average"` (no dependencies).

**Q: Bands too wide/narrow?**  
A: Width reflects actual historical volatility. Check data quality if bands seem incorrect.

---

For more details, see code comments in `tools/burnProcessing.py` or refer to the [Configuration Guide](CONFIGURATION.md).

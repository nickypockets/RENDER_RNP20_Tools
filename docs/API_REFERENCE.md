# API Reference

Complete reference for all public functions and classes in the Render Forward Guidance Tool.

## Table of Contents

- [Data Download (`tools.downloadData`)](#data-download-toolsdownloaddata)
- [Node Processing (`tools.nodeProcessing`)](#node-processing-toolsnodeprocessing)
- [Burn Processing (`tools.burnProcessing`)](#burn-processing-toolsburnprocessing)
- [OBhr Processing (`OBhrProcessing`)](#obhr-processing-obhrprocessing)
- [Policy Simulation (`policySimulation`)](#policy-simulation-policysimulation)

---

## Data Download (`tools.downloadData`)

### `download_data()`

Downloads data from Dune Analytics and external APIs.

```python
from tools.downloadData import download_data

download_data()
```

**Parameters:** None (uses configuration from `settings/download_ids.json` and `.env`)

**Returns:** None

**Side Effects:**
- Creates/updates `data/avail_data/avail.csv`
- Creates/updates `data/work_data/work.csv`
- Creates/updates `data/burns_data/burns.json`
- Creates/updates `data/OBhrs_data/OBhrs.json`

**Raises:**
- `EnvironmentError`: If `DUNE_API_KEY` not found in environment
- `FileNotFoundError`: If `settings/download_ids.json` not found
- `requests.RequestException`: If network requests fail

**Example:**

```python
# Download all data sources
download_data()

# Data is automatically merged with existing files
# Duplicates are removed based on (wallet, date, amount) keys
```

---

## Node Processing (`tools.nodeProcessing`)

### `aggregate_node_rewards()`

Aggregates node rewards from availability and work CSVs into a single summary.

```python
from tools.nodeProcessing import aggregate_node_rewards

df = aggregate_node_rewards(
    avail_csv=None,  # defaults to data/avail_data/avail.csv
    work_csv=None,   # defaults to data/work_data/work.csv
    output_csv=None  # defaults to data/node_summary.csv
)
```

**Parameters:**

- **`avail_csv`** (str | Path, optional): Path to availability CSV. If `None`, uses default path.
- **`work_csv`** (str | Path, optional): Path to work CSV. If `None`, uses default path.
- **`output_csv`** (str | Path, optional): Path for output CSV. If `None`, uses default path.

**Returns:**
- `pd.DataFrame`: Node summary with columns:
  - `Recipient Wallet`: Wallet address (lowercased)
  - `first_epoch`: First epoch the wallet appeared
  - `epoch_1`, `epoch_2`, ...: Reward amounts per epoch

**Side Effects:**
- Writes `output_csv` with aggregated data
- Creates `data/epoch_map.csv` with epoch-to-date mapping

**Example:**

```python
# Use defaults
df = aggregate_node_rewards()

# Custom paths
df = aggregate_node_rewards(
    avail_csv="custom/avail.csv",
    work_csv="custom/work.csv",
    output_csv="custom/output.csv"
)

# Access specific wallet data
wallet = "0x1234..."
wallet_data = df[df['Recipient Wallet'] == wallet.lower()]
print(wallet_data[['first_epoch', 'epoch_1', 'epoch_2']])
```

### `plot_sma_by_tier()`

Generates interactive chart showing node counts by tier over time with smoothing.

```python
from tools.nodeProcessing import plot_sma_by_tier

plot_sma_by_tier(use_dates=False)
```

**Parameters:**

- **`use_dates`** (bool, optional): 
  - `True`: Use month/year dates on X-axis (requires `epoch_map.csv`)
  - `False`: Use epoch numbers (default)

**Returns:** None

**Side Effects:**
- Creates `reports/node_sma_chart.html` (interactive Plotly chart)
- May create `reports/node_sma_chart.png` (static fallback if Plotly unavailable)
- Writes `reports/node_tiers.csv` with tier statistics

**Requirements:**
- `data/node_summary.csv` must exist (run `aggregate_node_rewards()` first)
- For date labels: `data/epoch_map.csv` must exist

**Example:**

```python
# Plot with epoch numbers
plot_sma_by_tier(use_dates=False)

# Plot with dates
plot_sma_by_tier(use_dates=True)
```

---

## Burn Processing (`tools.burnProcessing`)
### `create_burns_chart()`

Creates interactive chart of burn data with various decomposition options.

```python
from tools.burnProcessing import create_burns_chart

create_burns_chart(
    df=df_burns,
    use_dates=True,
    show_original_burns=False,
    show_sigma_bands=False,
    decomposition_method='savgol',
    sigma_levels=[1, 2, 3]
)
```

**Parameters:**

- **`df`** (pd.DataFrame, required): Burns DataFrame with required columns
- **`use_dates`** (bool, optional): Use dates vs epoch numbers. Default: `True`
- **`show_original_burns`** (bool, optional): Show raw burns line. Default: `False`
- **`show_sigma_bands`** (bool, optional): Show volatility bands. Default: `False`
- **`decomposition_method`** (str, optional): Method for trend/seasonality extraction:
  - `'statsmodels'`: Seasonal decomposition (most accurate, requires statsmodels) - **default**
  - `'moving_average'`: Simple moving average trend (fastest, most robust)
  - `'savgol'`: Savitzky-Golay filter (smooth trend, requires scipy)
  - Default: `'statsmodels'`
- **`sigma_levels`** (List[int], optional): Which sigma levels to display. Default: `[1, 2, 3]`

**Returns:** `go.Figure` (Plotly Figure object)

**Side Effects:**
- When called from command line (`python -m tools.burnProcessing`), saves to `reports/burns_chart.html`

**Requirements:**
- `data/burns_data/burns.json` must exist
- For `savgol`: scipy must be installed
- For `statsmodels`: statsmodels must be installed

**Example:**

```python
# Load and process data first
from tools.burnProcessing import df_burns, compute_sma_and_quarterly_growth

# Compute moving averages
df = compute_sma_and_quarterly_growth(df_burns, window=12)

# Create chart with Savitzky-Golay decomposition
fig = create_burns_chart(
    df=df,
    use_dates=True,
    show_original_burns=True,
    decomposition_method='savgol'
)

# Create chart with statsmodels seasonal decomposition and sigma bands
fig = create_burns_chart(
    df=df,
    decomposition_method='statsmodels',
    show_sigma_bands=True,
    sigma_levels=[1, 2, 3]
)

# Save or display
fig.write_html('custom_chart.html')
# or: fig.show()
```

---

## OBhr Processing (`OBhrProcessing`)

### `process()`

Processes OBhr liability data and generates aggregated statistics by tier.

```python
from OBhrProcessing import process

process(
    obhrs_path='data/OBhrs_data/OBhrs.json',
    node_summary_csv='data/node_summary.csv',
    out_csv='data/OBhrs_epoch_tier_totals.csv'
)
```

**Parameters:**

- **`obhrs_path`** (str): Path to OBhrs JSON file
- **`node_summary_csv`** (str): Path to node summary CSV (for tier classifications)
- **`out_csv`** (str): Path for output CSV

**Returns:** None

**Side Effects:**
- Writes `out_csv` with columns:
  - `epoch`: Epoch number
  - `T2_total`: Total T2 OBhrs
  - `T3_total`: Total T3 OBhrs
  - `unmapped_total`: OBhrs from unclassified wallets
- Creates `reports/obhrs_sma_chart.html` (interactive chart)
- Prints unmapped wallet report to console

**Example:**

```python
# Use defaults
process(
    obhrs_path='data/OBhrs_data/OBhrs.json',
    node_summary_csv='data/node_summary.csv',
    out_csv='data/OBhrs_epoch_tier_totals.csv'
)
```

### `create_obhrs_chart()`

Generates interactive OBhr trends chart.

```python
from OBhrProcessing import create_obhrs_chart

create_obhrs_chart(
    csv_path='data/OBhrs_epoch_tier_totals.csv',
    output_html='reports/obhrs_sma_chart.html',
    use_dates=False
)
```

**Parameters:**

- **`csv_path`** (str): Path to OBhr totals CSV
- **`output_html`** (str): Path for output HTML chart
- **`use_dates`** (bool, optional): Use dates vs epochs. Default: `False`

**Returns:** None

**Side Effects:**
- Creates interactive Plotly chart at `output_html`

---

## Policy Simulation (`policySimulation`)

### `PolicySimulation` Class

Main class for running policy simulations.

```python
from policySimulation import PolicySimulation

sim = PolicySimulation(
    burns_json=None,
    jobs_csv=None,
    avail_csv=None,
    tiers_csv=None,
    out_html=None,
    out_csv=None,
    out_classified=None,
    rebalance_period=None,
    first_rebalance_epoch=None,
    skip_node_summary_regen=False
)
```

**Constructor Parameters:**

- **`burns_json`** (Path, optional): Path to OBhrs JSON. Default: `data/OBhrs_data/OBhrs.json`
- **`jobs_csv`** (Path, optional): Path to work CSV. Default: `data/work_data/work.csv`
- **`avail_csv`** (Path, optional): Path to availability CSV. Default: `data/avail_data/avail.csv`
- **`tiers_csv`** (Path, optional): Path to node summary. Default: `data/node_summary.csv`
- **`out_html`** (Path, optional): Output HTML path. Default: `reports/policy_simulation.html`
- **`out_csv`** (Path, optional): Output CSV path. Default: `reports/policy_values.csv`
- **`out_classified`** (Path, optional): Classified wallets output. Default: `reports/tiered_wallets_classified.csv`
- **`rebalance_period`** (int, optional): Epochs between rebalances (4-24). Default: `12`
- **`first_rebalance_epoch`** (int, optional): First rebalance epoch. Default: `12`
- **`skip_node_summary_regen`** (bool, optional): Skip node processing if True. Default: `False`

**Class Attributes (configurable):**

```python
START_POLICY = 47435              # Initial policy value
REBALANCE_PERIOD = 12             # Epochs between rebalances
FIRST_REBALANCE_EPOCH = 12        # When rebalancing starts
F2, F3 = 2/3, 1/3                 # Tier weights
LAMBDA2 = 1.0                     # T2 sensitivity
LAMBDA3 = 1.0                     # T3 sensitivity
BAND_MULTIPLIER_UPPER = 3.0       # Upper band multiplier
BAND_MULTIPLIER_LOWER = 1.0       # Lower band multiplier
TRIGGER_ABOVE_UPPER = 1           # Trigger above upper band
NOCHANGE_BUFFER = 0.10            # No-change buffer (10%)
STEP_QUANTUM = 0.05               # Step size (5%)
ROUND_TO = 100                    # Rounding precision
```

### `sim.run()`

Runs the historical policy simulation.

```python
sim = PolicySimulation()
sim.run()
```

**Parameters:** None

**Returns:** None

**Side Effects:**
- Processes node data (unless `skip_node_summary_regen=True`)
- Classifies wallets into tiers
- Calculates policy values for all epochs
- Generates interactive chart
- Writes output files:
  - `reports/policy_simulation.html`
  - `reports/policy_values.csv`
  - `reports/tiered_wallets_classified.csv`

**Example:**

```python
# Basic run
sim = PolicySimulation()
sim.run()

# Fast run (skip node regeneration)
sim = PolicySimulation(skip_node_summary_regen=True)
sim.run()

# Custom cadence
sim = PolicySimulation(
    rebalance_period=8,
    first_rebalance_epoch=109
)
sim.run()
```

### `sim.run_scenario()`

Runs future policy scenarios with different growth curves. The tool includes 7 pre-configured educational scenarios that demonstrate policy behavior under various market conditions.

```python
sim.run_scenario(
    name="burn_exp",
    base="burn",
    curve="exponential",
    periods=52,
    multiplier=15,
    start_policy=15000.0
)
```

**Parameters:**

- **`name`** (str): Scenario name (used for output files)
- **`base`** (str): Base metric to project
  - `'burn'`: Project based on burn trends (demand-side)
  - `'node'`: Project based on node activity (supply-side)
  - `'hybrid'`: Project based on both burns and nodes
- **`curve`** (str): Growth curve type
  - `'linear'`: Linear growth (steady change)
  - `'exponential'`: Exponential growth (accelerating change)
  - `'s-curve'`: S-shaped growth curve (slow start, rapid middle, slow end)
  - `'bezier'`: Custom Bezier curve (requires schedule)
- **`periods`** (int): Number of future epochs to simulate
- **`multiplier`** (float | list): Growth multiplier
  - For linear/exponential/s-curve: single number representing final value relative to current
    - `1.5` = 50% increase
    - `0.4` = 60% decrease
    - `3.0` = 200% increase (3× current)
  - For bezier: list of (multiplier, duration) tuples
- **`start_policy`** (float): Starting policy value for scenario
- **`first_rebalance_epoch`** (int, optional): Absolute epoch for first rebalance (defaults to first future epoch)

**Returns:** 
- `tuple`: (df_sim, steps, meta)
  - `df_sim` (pd.DataFrame): Full simulation results
  - `steps` (tuple): (x_values, y_values) for step plot
  - `meta` (pd.DataFrame): Decision metadata for each epoch

**Side Effects:**
- Creates `reports/future_sims/{name}.html` - Interactive visualization
- Creates `reports/future_sims/{name}.csv` - Detailed data
- Creates `reports/future_sims/{name}_decisions.csv` - Decision log

**Pre-Configured Scenarios:**

The tool includes 7 ready-to-use scenarios (commented at the end of `policySimulation.py`). **Scenarios 2 and 3 are featured in the Render Network Proposal (RNP)** as primary examples:

**Burn-Based Scenarios (Demand-Driven):**

1. **steady_growth** - Healthy network growth (50% increase over 52 epochs)
   - Demonstrates gradual policy increases with 10% cap
   
2. **market_downturn** ⭐ **Featured in RNP** - Bear market scenario (60% decline over 52 epochs)
   - Shows controlled decreases and no-change buffer protection
   - Example of how policy protects nodes during downturns
   
3. **explosive_growth** ⭐ **Featured in RNP** - Viral adoption (3× increase over 36 epochs)
   - Shows rapid but controlled increases ensuring burns exceed issuance
   - Demonstrates importance of 10% cap and cap escalation

**Node-Based Scenarios (Supply-Driven):**

4. **node_expansion** - Growing operator base (50% increase over 52 epochs)
   - Shows node multiplier component in action
   
5. **node_attrition** - Operator exodus (35% decline over 52 epochs)
   - Demonstrates adaptive response to capacity constraints
   
6. **tier_migration** - Quality upgrade (T2→T3 shift over 40 epochs)
   - Uses `run_hypothetical()` with tier-specific multipliers
   - Shows equal-pain factor and dynamic tier weights
   
7. **activity_surge** - Increased utilization (2× hours over 36 epochs)
   - Demonstrates hours-change multiplier component

**Examples:**

```python
sim = PolicySimulation(skip_node_summary_regen=True)

# RNP Featured Scenario 2: Market Downturn
# Shows how policy protects nodes during 60% burn decline
sim.run_scenario(
    name="market_downturn",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=0.4,  # 60% decline
    start_policy=15000.0
)

# RNP Featured Scenario 3: Explosive Growth  
# Shows how policy handles 200% burn increase
sim.run_scenario(
    name="explosive_growth",
    base="burn",
    curve="exponential",
    periods=36,
    multiplier=3.0,  # 3× current burns
    start_policy=15000.0
)

# Steady growth scenario
sim.run_scenario(
    name="steady_growth",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=1.5,  # 50% increase
    start_policy=15000.0
)

# Node-based scenario
sim.run_scenario(
    name="node_expansion",
    base="node",
    curve="linear",
    periods=52,
    multiplier=1.5,
    start_policy=15000.0
)

# Custom schedule with Bezier curve
# Format: [(multiplier, duration), ...]
schedule = [(1.5, 25), (0.7, 52), (0.3, 20)]
sim.run_scenario(
    name="hybrid_bezier",
    base="burn",
    curve="bezier",
    periods=97,
    multiplier=schedule,
    start_policy=15000.0
)
```

**Using the Pre-Configured Scenarios:**

All 7 scenarios are documented with full explanations at the end of `policySimulation.py`. To run them:

1. Open `policySimulation.py`
2. Scroll to the bottom (after `if __name__ == "__main__":`)
3. Find the scenario you want (e.g., "Market Downturn")
4. Uncomment the `sim.run_scenario()` call
5. Run: `python policySimulation.py`
6. View results in `reports/future_sims/`

**Understanding Results:**

Each scenario output shows:
- **Policy Evolution**: How the policy level changes over time
- **Rebalance Points**: When decisions occur (marked with diamonds)
- **Pay Bands**: 1× (lower) to 3× (upper) anchor boundaries
- **Growth Patterns**: How burns/nodes evolve according to the curve
- **Decision Logic**: Why each change was made (in CSV files)

The interactive HTML charts allow you to:
- Hover over points to see exact values
- Zoom into specific time periods
- Toggle traces on/off
- Export as PNG images

### `sim.run_hypothetical()`

Runs a hypothetical simulation applying custom multipliers to nodes and hours from a specific epoch onward. Useful for tier-specific scenarios (like tier migration).

```python
df, steps, meta = sim.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier={"T2": 0.7, "T3": 1.4},
    hours_multiplier={"T2": 0.7, "T3": 1.4}
)
```

**Parameters:**

- **`start_epoch`** (int): Epoch to start applying multipliers from
- **`start_policy`** (float): Starting policy value
- **`nodes_multiplier`** (float | dict, optional): 
  - Single float: applies to both T2 and T3 nodes
  - Dict: `{"T2": multiplier, "T3": multiplier}` for tier-specific changes
- **`hours_multiplier`** (float | dict, optional):
  - Single float: applies to both T2 and T3 hours
  - Dict: `{"T2": multiplier, "T3": multiplier}` for tier-specific changes

**Returns:**
- `tuple`: (df_sim, steps, meta) - same structure as `run_scenario()`

**Side Effects:**
- Writes `reports/policy_values_hypothetical.csv`
- Writes `reports/policy_values_hypothetical_decisions.csv`

**Examples:**

```python
sim = PolicySimulation(skip_node_summary_regen=True)

# Tier Migration: T2 nodes decline 30%, T3 nodes grow 40%
df, steps, meta = sim.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier={"T2": 0.7, "T3": 1.4},
    hours_multiplier={"T2": 0.7, "T3": 1.4}
)

# Uniform node growth: both tiers grow 50%
df, steps, meta = sim.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier=1.5
)

# Activity surge: hours double but node count stays same
df, steps, meta = sim.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    hours_multiplier=2.0
)
```

**Use Case - Scenario 6 (Tier Migration):**

This method is ideal for simulating compositional changes in the node network, such as when operators upgrade from T2 to T3 nodes:

```python
# Simulate 10 months of T2→T3 migration
sim_tier = PolicySimulation(skip_node_summary_regen=True)
df, steps, meta = sim_tier.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier={"T2": 0.7, "T3": 1.4}  # T2 -30%, T3 +40%
)
# Results show how equal-pain factor and tier weights adapt
```

### `sim.explain_latest_policy()`

**This is the main function you need** - it tells you what the policy should be for the current epoch.

```python
result = sim.explain_latest_policy(current_policy=15000)
print(result['reason'])
print(f"\nRecommended Policy: {result['new_policy']:,.0f}")
```

**Parameters:**

- **`current_policy`** (float, optional): The currently active policy level
  - Use this to specify what the current policy is
  - If omitted, uses the last historical policy value from simulation

**Returns:**
- `dict` containing:
  - **`new_policy`** (float): **The recommended policy value** - this is what you need!
  - `epoch` (int): The epoch being analyzed
  - `is_rebalance_epoch` (bool): Whether this is a rebalance epoch
  - `decision` (str): Type of decision made
  - `reason` (str): **Complete explanation** of the calculation including:
    - Burn averages and growth rates
    - Node multiplier with tier breakdown
    - Equal-pain factor
    - Anchor value
    - Cap application
    - Final policy recommendation
  - `details` (dict): All intermediate calculations

**Examples:**

```python
sim = PolicySimulation()

# Get recommendation with current policy level
result = sim.explain_latest_policy(current_policy=15000)

# The most important values:
print(f"New Policy: {result['new_policy']:,.0f}")  # e.g., 16,500
print(f"\nExplanation:\n{result['reason']}")

# Check if it's a rebalance epoch
if result['is_rebalance_epoch']:
    print(f"\nThis IS a rebalance epoch")
else:
    print(f"\nThis is NOT a rebalance epoch")
```

**Example Output:**

```
Epoch 96 average burns over the last 12 epochs were 12,562 tokens (-9.10% growth).
Node multiplier 1.354 uses Tier mix f2 +58.87%, f3 +41.13%, node growth T2 +43.18%,
T3 +25.62%, hours change T2 +16.20%, T3 +31.92%.
Equal-pain factor 0.989 produces an anchor of 15,289 tokens.
Recommend adjusting policy to 12,100 (+10.00% step).

New Policy: 12,100
```

**Important Notes:**
- This method analyzes the **latest available epoch** in your data
- It automatically checks if it's a rebalance epoch (based on the configured cadence)
- The output includes the **complete calculation** showing how the recommendation was derived
- If the previous decision was to increase by 10%, and this method suggests another 10% increase, it will correctly show the compounded result

---

## Utility Functions

### `apply_reciprocal()`

Applies reciprocal fraction logic for policy adjustments.

```python
from policySimulation import PolicySimulation

# This is an internal method but can be used for calculations
sim = PolicySimulation()
new_policy = sim.apply_reciprocal(
    prev_policy=15000,
    r_step=-0.10
)
print(new_policy)  # 16363.636... (15000 / 0.90)
```

**Parameters:**

- **`prev_policy`** (float): Previous policy value
- **`r_step`** (float): Adjustment step as decimal
  - Positive values: multiply by (1 + r_step)
  - Negative values: divide by (1 - r_step)

**Returns:**
- `float`: New policy value

**Examples:**

```python
# Increase by 10%
apply_reciprocal(1000, 0.10)  # Returns: 1100.0

# Decrease by 10%
apply_reciprocal(1000, -0.10)  # Returns: 1111.111... (1000 / 0.9)

# Symmetry test
val = apply_reciprocal(1000, 0.10)   # 1100
val = apply_reciprocal(val, -0.10)   # 1000 (back to original)
```

---

## Error Handling

All functions may raise standard Python exceptions:

- **`FileNotFoundError`**: Required input file doesn't exist
- **`ValueError`**: Invalid parameter values
- **`KeyError`**: Missing expected columns in data
- **`EnvironmentError`**: Missing environment variables

Always ensure:
1. `.env` file exists with `DUNE_API_KEY`
2. Required data files are downloaded
3. Processing steps are run in correct order

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for solutions to common errors.

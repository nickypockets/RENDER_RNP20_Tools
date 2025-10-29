# Configuration Guide

This guide covers all configuration options for the Render Forward Guidance Tool.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Query Configuration](#query-configuration)
- [Simulation Parameters](#simulation-parameters)
- [Chart Configuration](#chart-configuration)
- [Data Paths](#data-paths)

## Environment Variables

### `.env` File

Create a `.env` file in the project root with your Dune API credentials:

```env
# Required: Your Dune Analytics API key
DUNE_API_KEY=your_api_key_here

# Optional: Override default query timeout (seconds)
# DUNE_TIMEOUT=300
```

### Getting Your Dune API Key

1. Sign up at [dune.com](https://dune.com) (free tier works)
2. Navigate to [dune.com/settings/api](https://dune.com/settings/api)
3. Click "Create new API key"
4. Copy the key and paste it into your `.env` file

**Security Note:** Never commit your `.env` file to version control. It's already included in `.gitignore`.

## Query Configuration

### `settings/download_ids.json`

Configure which Dune queries and data sources to use:

```json
{
  "avail": 4833735,
  "work": 3456781,
  "burns": "https://stats.renderfoundation.com/epoch-burn-stats-data.json",
  "OBhrs": "https://stats.renderfoundation.com/liability-epochs-data.json"
}
```

**Fields:**

- **`avail`** (integer): Dune query ID for availability rewards
  - Default query tracks node availability payments
  - Must return columns: Recipient Wallet, Date, $RENDER Rewards, etc.

- **`work`** (integer): Dune query ID for work rewards
  - Default query tracks node work payments
  - Must return columns: Recipient Wallet, Date, $RENDER Rewards, etc.

- **`burns`** (string URL): JSON endpoint for burn statistics
  - Must return data with epoch IDs and burn amounts
  - Expected format: `{"data": [{"id": 1, "burned": 1000, ...}, ...]}`

- **`OBhrs`** (string URL): JSON endpoint for OBhr liability data
  - Must return data with epoch and liability information
  - Expected format: `{"data": [{"epochId": 1, "walletAddress": "0x...", ...}, ...]}`

### Using Custom Queries

To use your own Dune queries:

1. Create or fork a query on Dune
2. Copy the query ID from the URL (e.g., `dune.com/queries/1234567` → ID is `1234567`)
3. Update `settings/download_ids.json` with your query ID
4. Ensure your query returns the expected columns:
   - `Recipient Wallet` or similar wallet identifier
   - `Date` or `block_time` for timestamps
   - `$RENDER Rewards` or `amount` for reward values

## Simulation Parameters

### Policy Simulation Configuration

Edit `policySimulation.py` to customize simulation behavior:

```python
class PolicySimulation:
    # Starting conditions
    START_POLICY = 47435              # Initial policy value (RENDER)
    REBALANCE_PERIOD = 12             # Epochs between rebalances
    FIRST_REBALANCE_EPOCH = 12        # First epoch to apply rebalancing
    
    # Tier weights for anchor calculation
    F2 = 2 / 3                        # T2 weight (66.67%)
    F3 = 1 / 3                        # T3 weight (33.33%)
    
    # Sensitivity parameters
    LAMBDA2 = 1.0                     # T2 sensitivity multiplier
    LAMBDA3 = 1.0                     # T3 sensitivity multiplier
    
    # Band multipliers (sigma bands)
    BAND_MULTIPLIER_UPPER = 3.0       # Upper band = anchor × 3.0
    BAND_MULTIPLIER_LOWER = 1.0       # Lower band = anchor × 1.0
    
    # Trigger configuration
    TRIGGER_ABOVE_UPPER = 1           # Triggers above upper band
    
    # Step and buffer configuration
    NOCHANGE_BUFFER = 0.10            # No change if within ±10% of anchor
    STEP_QUANTUM = 0.05               # Each step is ±5%
    ROUND_TO = 100                    # Round policy values to nearest 100
    
    # Exchange rate
    DEFAULT_EURUSD = 1.0              # EUR/USD rate (if needed)
```

### Rebalance Period Constraints

The `rebalance_period` must be between 4 and 24 epochs:

```python
# Valid examples
sim = PolicySimulation(rebalance_period=4)   # Rebalance every 4 epochs
sim = PolicySimulation(rebalance_period=12)  # Default: every 12 epochs
sim = PolicySimulation(rebalance_period=24)  # Maximum: every 24 epochs

# Invalid - will raise ValueError
sim = PolicySimulation(rebalance_period=3)   # Too short
sim = PolicySimulation(rebalance_period=25)  # Too long
```

### Cap Escalation Logic

The simulation automatically applies cap escalation:

- **First consecutive rebalance** in same direction: 10% cap
- **Second consecutive rebalance** in same direction: 20% cap
- **Third consecutive rebalance** in same direction: 30% cap
- **Direction change or no-change**: Reset cap to 10%

Example:
```
Epoch 12: +10% increase (capped at 10%)
Epoch 24: +15% increase (capped at 20% because 2nd consecutive up)
Epoch 36: -8% decrease (capped at 10% because direction changed)
```

### Reciprocal Fractions

The simulation uses true reciprocal fractions for decreases:

- **Increase by 10%**: Multiply by 1.10
- **Decrease by 10%**: Divide by 1.10 (equivalent to multiplying by 0.909...)

This ensures symmetry: increasing then decreasing by the same percentage returns to the original value.

## Chart Configuration

### Date vs Epoch Display

All chart functions support `use_dates` parameter:

```python
# Use epoch numbers (default for most functions)
create_burns_chart(use_dates=False)
plot_sma_by_tier(use_dates=False)

# Use month/year dates (requires epoch_map.csv)
create_burns_chart(use_dates=True)
plot_sma_by_tier(use_dates=True)
```

**Requirements for date display:**
- `data/epoch_map.csv` must exist (generated by `nodeProcessing`)
- `burns.json` must contain valid date information

### Burn Chart Options

```python
from tools.burnProcessing import create_burns_chart

create_burns_chart(
    df=df_burns,                         # Your burns DataFrame
    use_dates=True,                      # X-axis: dates vs epochs
    show_original_burns=False,           # Show raw burns line (default: hidden)
    show_sigma_bands=False,              # Show volatility bands
    decomposition_method='savgol',       # 'statsmodels', 'moving_average', 'savgol'
    sigma_levels=[1, 2, 3]               # Which sigma bands to display
)
```

**Decomposition methods:**
- **`statsmodels`**: Seasonal decomposition (most accurate, requires statsmodels) - **default**
- **`moving_average`**: Simple moving average trend (fastest, most robust)
- **`savgol`**: Savitzky-Golay filter (smooth trend, requires scipy)

**Note:** The actual chart generation happens internally. To generate a chart from command line, use:
```bash
python -m tools.burnProcessing
```

This will load the data, process it, and save the chart to `reports/burns_chart.html`.

### Policy Chart Configuration

The policy simulation chart automatically includes:
- Policy line (actual computed values)
- Upper band (3× anchor by default)
- Lower band (1× anchor by default)
- Anchor line (calculated from T2/T3 OBhrs)
- Rebalance markers

**Removed lines** (previously redundant):
- Trigger line (was duplicate of upper band)
- Anchor pre-logic line (was duplicate of lower band)

## Data Paths

### Default Data Structure

```
data/
├── avail_data/
│   └── avail.csv                 # Node availability rewards
├── work_data/
│   └── work.csv                  # Node work rewards
├── burns_data/
│   └── burns.json                # Burn statistics
├── OBhrs_data/
│   └── OBhrs.json                # OBhr liability data
├── node_summary.csv              # Processed node summary
├── epoch_map.csv                 # Epoch-to-date mapping
└── OBhrs_epoch_tier_totals.csv   # Aggregated OBhr data
```

### Custom Data Paths

You can specify custom paths when running functions:

#### Node Processing

```python
from tools.nodeProcessing import aggregate_node_rewards

df = aggregate_node_rewards(
    avail_csv="path/to/custom_avail.csv",
    work_csv="path/to/custom_work.csv",
    output_csv="path/to/custom_output.csv"
)
```

#### Policy Simulation

```python
from policySimulation import PolicySimulation

sim = PolicySimulation(
    burns_json="path/to/burns.json",
    jobs_csv="path/to/work.csv",
    avail_csv="path/to/avail.csv",
    tiers_csv="path/to/node_summary.csv",
    out_html="path/to/output.html",
    out_csv="path/to/output.csv",
    out_classified="path/to/classified.csv"
)
sim.run()
```

#### OBhr Processing

```python
from OBhrProcessing import process

process(
    obhrs_path='path/to/OBhrs.json',
    node_summary_csv='path/to/node_summary.csv',
    out_csv='path/to/output.csv'
)
```

### Path Portability

All scripts now use file-relative paths (as of recent updates), meaning:

✅ Works when run from project root: `python policySimulation.py`
✅ Works when run from subdirectory: `cd tools && python -m downloadData`
✅ Works when imported as module: `from tools import downloadData`
✅ Works on all operating systems (Windows, macOS, Linux)

No need to worry about current working directory or execution location.

## Advanced Configuration

### Performance Optimization

**Skip Node Summary Regeneration:**

If you're running multiple simulations and haven't changed the raw data:

```python
sim = PolicySimulation(skip_node_summary_regen=True)
sim.run()
```

This can save significant time by reusing the existing `node_summary.csv`.

### Multiple Simulations

Run different scenarios in sequence:

```python
sim = PolicySimulation()

# Base simulation
sim.run()

# Future scenarios
sim.run_scenario(
    name="burn_exp",
    base="burn",
    curve="exponential",
    periods=52,
    multiplier=15,
    start_policy=15000.0
)

sim.run_scenario(
    name="node_scurve",
    base="node",
    curve="s-curve",
    periods=52,
    multiplier=1.3,
    start_policy=15000.0
)
```

### Custom Rebalance Cadence

Test different rebalancing frequencies:

```python
# Aggressive: rebalance every 4 epochs
sim_fast = PolicySimulation(
    rebalance_period=4,
    first_rebalance_epoch=4
)
sim_fast.run()

# Conservative: rebalance every 24 epochs
sim_slow = PolicySimulation(
    rebalance_period=24,
    first_rebalance_epoch=24
)
sim_slow.run()
```

## Troubleshooting Configuration

### Common Issues

**Problem:** "DUNE_API_KEY not found"
- **Solution:** Create `.env` file in project root with your API key

**Problem:** "FileNotFoundError: data/node_summary.csv"
- **Solution:** Run `python -m tools.nodeProcessing` first

**Problem:** "ValueError: rebalance_period must be between 4 and 24"
- **Solution:** Adjust rebalance_period to valid range

**Problem:** Charts show epochs instead of dates
- **Solution:** Set `use_dates=True` and ensure `data/epoch_map.csv` exists

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

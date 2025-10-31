# Render Forward Guidance Tool

A comprehensive data analysis and policy simulation toolkit for the Render Network. This tool downloads node operation data from Dune Analytics and the  [Render stats webpage](https://stats.renderfoundation.com/), processes historical performance metrics, and runs policy simulations to model future issuance behavior.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Download Data](#1-download-data)
  - [2. Process Data](#2-process-data)
  - [3. Run Simulations](#3-run-simulations)
- [Command Line Reference](#command-line-reference)
- [Future Scenario Projections](#future-scenario-projections)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/nickypockets/Render-Forward-Guidance-Tool.git
cd Render-Forward-Guidance-Tool

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Dune API
# Create .env file with your Dune API key (see Configuration section)

# 5. Download data
python -m tools.downloadData

# 6. Process data (order matters!)
python -m tools.nodeProcessing
python OBhrProcessing.py
python -m tools.burnProcessing

# 7. Get policy recommendation (MAIN USE CASE)
python
>>> from policySimulation import PolicySimulation
>>> sim = PolicySimulation()
>>> result = sim.explain_latest_policy(current_policy=15000)
>>> print(result['reason'])
>>> print(f"\nRecommended Policy: {result['new_policy']:,.0f}")

# Optional: Run full historical simulation
python policySimulation.py
```

**What you really need:** After downloading and processing data, use `explain_latest_policy()` with your current policy level - it tells you what the new policy should be with a complete explanation.

## Prerequisites

- **Python 3.11+** (Python 3.10 may work but 3.11+ is recommended)
- **Dune Analytics Account** (free tier works)
- **Git** (for cloning the repository)

## Installation

### 1. Install Python

If you don't have Python installed:

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

**macOS:**
```bash
brew install python@3.11
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### 2. Clone the Repository

```bash
git clone https://github.com/nickypockets/Render-Forward-Guidance-Tool.git
cd Render-Forward-Guidance-Tool
```

### 3. Create Virtual Environment

A virtual environment keeps dependencies isolated:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# macOS/Linux:
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt when activated.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `plotly` - Interactive visualizations
- `requests` - HTTP requests
- `dune-client` - Dune Analytics API
- `python-dotenv` - Environment variable management
- `statsmodels` - Statistical modeling (optional, for advanced burn processing)
- `scipy` - Scientific computing (optional, for advanced burn processing)

## Configuration

### Setting Up Dune Analytics

1. **Create a Dune Account**
   - Go to [dune.com](https://dune.com)
   - Click "Sign Up" (free tier is sufficient)
   - Verify your email

2. **Get Your API Key**
   - Log in to Dune
   - Go to [dune.com/settings/api](https://dune.com/settings/api)
   - Click "Create new API key"
   - Copy the generated key

3. **Create `.env` File**

   In the project root directory, create a file named `.env`:

   ```bash
   # Windows (PowerShell):
   New-Item .env -ItemType File

   # macOS/Linux:
   touch .env
   ```

   Open `.env` in a text editor and add:

   ```
   DUNE_API_KEY=your_api_key_here
   ```

   Replace `your_api_key_here` with your actual Dune API key.

   **Important:** Never commit your `.env` file to Git. It's already in `.gitignore`.

### Query Configuration

The project uses Dune queries configured in `settings/download_ids.json`:

```json
{
  "avail": 4833735,
  "work": 3456781,
  "burns": "https://stats.renderfoundation.com/epoch-burn-stats-data.json",
  "OBhrs": "https://stats.renderfoundation.com/liability-epochs-data.json"
}
```

- `avail` and `work` are Dune query IDs
- `burns` and `OBhrs` are direct JSON URLs from Render Foundation

You can modify these IDs to use different Dune queries if needed.

## Usage

### 1. Download Data

Download the latest data from Dune Analytics and Render Foundation:

```bash
python -m tools.downloadData
```

**What it does:**
- Fetches availability rewards from Dune (query ID: 4833735)
- Fetches work rewards from Dune (query ID: 3456781)
- Downloads burn statistics from Render Foundation
- Downloads OBhr (liability) data from Render Foundation
- Saves all data to `data/` folder

**Output files:**
- `data/avail_data/avail.csv` - Node availability rewards
- `data/work_data/work.csv` - Node work rewards
- `data/burns_data/burns.json` - Burn statistics per epoch
- `data/OBhrs_data/OBhrs.json` - OBhr liability data per epoch

**Incremental updates:** The script merges new data with existing files, so you can run it multiple times to update without losing historical data.

### 2. Process Data

Process raw data into analysis-ready formats. 

**⚠️ Processing Order is Critical:**
1. **Node Processing** (creates `node_summary.csv`)
2. **OBhr Processing** (requires `node_summary.csv` for tier classifications)
3. **Burn Processing** (required for policy simulations)

#### Process Node Data (Run First)

```bash
python -m tools.nodeProcessing
```

**What it does:**
- Reads `data/avail_data/avail.csv` and `data/work_data/work.csv`
- Aggregates rewards per wallet per epoch
- Classifies nodes into tiers (T2/T3) based on reward patterns
- Creates epoch-to-date mapping from burn data
- Generates node activity chart

**Output files:**
- `data/node_summary.csv` - One row per wallet with all epochs **(required by OBhrProcessing)**
- `data/epoch_map.csv` - Epoch number to date mapping
- `reports/node_sma_chart.html` - Interactive tier activity chart
- `reports/node_tiers.csv` - Tier classifications

#### Process OBhr Data (Run Second)

```bash
python OBhrProcessing.py
```

**What it does:**
- Reads `data/OBhrs_data/OBhrs.json`
- **Requires `data/node_summary.csv`** for tier classifications (from nodeProcessing)
- Aggregates OBhr totals by epoch and tier (T2/T3)
- Generates interactive chart showing OBhr trends

**Output files:**
- `data/OBhrs_epoch_tier_totals.csv` - OBhr totals by epoch and tier
- `reports/obhrs_sma_chart.html` - Interactive visualization

#### Process Burns Data (Run Third - Required for Simulations)

```bash
python -m tools.burnProcessing
```

**What it does:**
- Reads `data/burns_data/burns.json`
- Calculates smoothed burn trends and growth rates
- Generates interactive chart with smoothing options
- **Computes quarterly growth metrics used by policy simulations**

**Output files:**
- `reports/burns_chart.html` - Interactive burn trends visualization

**Chart options:**
- Use dates or epoch numbers on X-axis
- Show/hide original burns line
- Display sigma bands for volatility analysis

**Important:** This step is required before running policy simulations as it calculates the burn growth rates that the policy mechanism uses.

### 3. Run Simulations

#### Get the Latest Policy Recommendation (Main Use Case)

**This is what you need most of the time** - it tells you what the policy should be for the current epoch:

```python
from policySimulation import PolicySimulation

sim = PolicySimulation()
result = sim.explain_latest_policy(current_policy=15000)

print(result['reason'])
print(f"\nRecommended Policy: {result['new_policy']:,.0f} tokens")
```

**What it does:**
- Analyzes the most recent epoch data
- Calculates what the policy level should be
- **Prints a complete explanation with the recommended policy value**
- Shows all the math: anchor calculation, cap application, and final value

**Parameters:**
- `current_policy` (float): The currently active policy level (e.g., 15000)

**Output includes:**
- Burn averages and growth rates
- Node multiplier calculations with tier mix (T2/T3)
- Equal-pain factor calculation
- Anchor value (the target based on burns and node activity)
- Step calculation with any caps applied
- **Final recommended policy value**

**Example output:**
```
Epoch 96 average burns over the last 12 epochs were 12,562 tokens (-9.10% growth).
Node multiplier 1.354 uses Tier mix f2 +58.87%, f3 +41.13%, node growth T2 +43.18%,
T3 +25.62%, hours change T2 +16.20%, T3 +31.92%.
Equal-pain factor 0.989 produces an anchor of 15,289 tokens.
Recommend adjusting policy to 12,100 (+10.00% step).

Recommended Policy: 12,100 tokens
```

#### Run Historical Simulation (Optional)

If you want to see the full historical simulation with visualizations:

```bash
python policySimulation.py
```

**What it does:**
- Loads historical OBhr, node work, and availability data
- Simulates policy adjustments across all historical epochs
- Calculates anchor values and step changes
- Applies cap escalation for consecutive rebalances
- Generates interactive visualization with policy bands

**Output files:**
- `reports/policy_simulation.html` - Interactive simulation chart
- `reports/policy_values.csv` - Policy values per epoch
- `reports/policy_decisions.csv` - Decision log for each rebalance epoch
- `reports/tiered_wallets_classified.csv` - Wallet tier classifications

**Key simulation parameters (configurable in code):**
- `START_POLICY` - Initial policy value (default: 47435)
- `REBALANCE_PERIOD` - Epochs between rebalances (default: 12)
- `FIRST_REBALANCE_EPOCH` - When rebalancing starts (default: 12)
- `BAND_MULTIPLIER_UPPER` - Upper band multiplier (default: 3.0)
- `BAND_MULTIPLIER_LOWER` - Lower band multiplier (default: 1.0)

## Command Line Reference

### Download Data

```bash
# Download all data sources
python -m tools.downloadData

# The script automatically:
# - Checks for existing data
# - Performs incremental updates
# - Handles duplicate removal
# - Adds epoch_end timestamps
```

### Process OBhrs

```bash
# Process OBhr data with default settings
python OBhrProcessing.py

# With custom parameters (modify in code):
# - use_dates: Show dates instead of epoch numbers
# - Custom input/output paths
```

### Process Nodes

```bash
# Process node data with defaults
python -m tools.nodeProcessing

# Custom usage from Python:
from tools.nodeProcessing import aggregate_node_rewards, plot_sma_by_tier

# Process with custom paths
df = aggregate_node_rewards(
    avail_csv="path/to/avail.csv",
    work_csv="path/to/work.csv",
    output_csv="path/to/output.csv"
)

# Generate chart with date labels
plot_sma_by_tier(use_dates=True)
```

### Get Policy Recommendation

**Most Common Usage** - Get the current policy recommendation:

```python
from policySimulation import PolicySimulation

sim = PolicySimulation()

# Pass the current active policy level
result = sim.explain_latest_policy(current_policy=15000)

# The result tells you what the policy should be
print(result['reason'])  # Detailed explanation
print(f"\nNew Policy: {result['new_policy']:,.0f}")  # The recommended value
```

**What you get:**
- Complete explanation of the calculation
- Burn trends and growth rates
- Node activity changes (T2 and T3)
- The anchor value
- **The recommended policy level** - this is what matters!

### Run Policy Simulation

```bash
# Run with default settings
python policySimulation.py

# Custom usage from Python:
from policySimulation import PolicySimulation

# Get latest policy recommendation with current policy level
sim = PolicySimulation()
result = sim.explain_latest_policy(current_policy=15000)
print(f"Recommended: {result['new_policy']:,.0f} tokens")

# Run full historical simulation
sim.run()

# Run with custom rebalance cadence
sim = PolicySimulation(
    rebalance_period=8,  # Rebalance every 8 epochs
    first_rebalance_epoch=109,  # Start at epoch 109
    skip_node_summary_regen=True  # Skip node processing if already done
)
sim.run()

# Run future scenarios
sim.run_scenario(
    name="burn_exp",
    base="burn",
    curve="exponential",
    periods=52,
    multiplier=15,
    start_policy=15000.0
)
```

### Process Burns (Advanced)

```bash
# Process burns with smoothing
python -m tools.burnProcessing

# From Python with options:
from tools.burnProcessing import create_burns_chart

create_burns_chart(
    df=df_burns,  # Your burns DataFrame
    use_dates=True,  # Use month/year labels
    show_original_burns=True,  # Show raw data line
    show_sigma_bands=True,  # Show volatility bands
    decomposition_method='savgol'  # or 'statsmodels', 'moving_average'
)
```

## Project Structure

```
Render-Forward-Guidance-Tool/
├── .env                          # Your Dune API key (create this, not in Git)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
│
├── data/                         # Downloaded and processed data
│   ├── avail_data/avail.csv
│   ├── work_data/work.csv
│   ├── burns_data/burns.json
│   ├── OBhrs_data/OBhrs.json
│   ├── node_summary.csv          # Processed node data
│   ├── epoch_map.csv             # Epoch-to-date mapping
│   └── OBhrs_epoch_tier_totals.csv
│
├── reports/                      # Generated charts and analysis
│   ├── policy_simulation.html
│   ├── policy_values.csv
│   ├── obhrs_sma_chart.html
│   ├── node_sma_chart.html
│   ├── burns_chart.html
│   └── tiered_wallets_classified.csv
│
├── settings/                     # Configuration
│   └── download_ids.json         # Dune query IDs and data URLs
│
├── tools/                        # Processing utilities
│   ├── __init__.py
│   ├── downloadData.py           # Data fetching from Dune/APIs
│   ├── nodeProcessing.py         # Node reward aggregation
│   └── burnProcessing.py         # Burn smoothing and analysis
│
├── docs/                         # Additional documentation
│   ├── CONFIGURATION.md
│   ├── API_REFERENCE.md
│   └── TROUBLESHOOTING.md
│
├── OBhrProcessing.py             # OBhr data processing script
└── policySimulation.py           # Policy simulation engine
```

## Documentation

Additional documentation is available in the `docs/` folder:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 15 minutes
- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration options
- **[API Reference](docs/API_REFERENCE.md)** - Function and class documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Future Scenarios](docs/SCENARIOS.md)** - Complete guide to the 7 scenario projections ⭐

### Advanced Topics

- **[Sigma Bands](docs/SIGMA_BANDS.md)** - Statistical deviation bands for burn analysis

## Workflow Example

Here's a complete workflow from start to finish:

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Download latest data (run weekly or as needed)
python -m tools.downloadData

# 3. Process all data (required before running simulations - order matters!)
python -m tools.nodeProcessing
python OBhrProcessing.py
python -m tools.burnProcessing

# 4. Run simulation
python policySimulation.py

# 5. View results
# Open reports/policy_simulation.html in your browser
# Open reports/obhrs_sma_chart.html in your browser
# Open reports/node_sma_chart.html in your browser
```

## Customizing Simulations

To customize simulation parameters, edit `policySimulation.py`:

```python
# Find the PolicySimulation class and modify these constants:
START_POLICY = 47435              # Starting policy value
REBALANCE_PERIOD = 12             # Epochs between rebalances (4-24)
FIRST_REBALANCE_EPOCH = 12        # When to start rebalancing
BAND_MULTIPLIER_UPPER = 3.0       # Upper band (3× anchor)
BAND_MULTIPLIER_LOWER = 1.0       # Lower band (1× anchor)
STEP_QUANTUM = 0.05               # Adjustment step size (5%)
NOCHANGE_BUFFER = 0.10            # No-change buffer (10%)
```

## Future Scenario Projections

The tool includes 7 pre-configured future scenarios that demonstrate how the policy responds to different market conditions. These scenarios are **featured in the Render Network Proposal (RNP)** and serve as educational examples.

### Running Future Scenarios

At the end of `policySimulation.py`, you'll find commented examples of all 7 scenarios. Simply uncomment the scenario you want to run:

```python
# Open policySimulation.py and scroll to the bottom
# Uncomment any scenario to run it

# Example: Run the "Market Downturn" scenario (featured in RNP)
sim.run_scenario(
    name="market_downturn", 
    base="burn", 
    curve="linear", 
    periods=52, 
    multiplier=0.4, 
    start_policy=15000.0
)
```

### Scenario Categories

**Burn-Based Scenarios (Demand-Driven):**

These show how the policy responds to changes in network usage and token burns:

1. **Steady Growth** - Healthy 50% growth over 1 year
2. **Market Downturn** ⭐ *Featured in RNP* - 60% decline simulating a bear market
3. **Explosive Growth** ⭐ *Featured in RNP* - 3× increase simulating viral adoption

**Node-Based Scenarios (Supply-Driven):**

These show how the policy responds to changes in node operator participation:

4. **Node Expansion** - 50% increase in node capacity over 1 year
5. **Node Attrition** - 35% decline in active operators
6. **Tier Migration** - Shift from T2 to T3 nodes (quality upgrade)
7. **Activity Surge** - Existing nodes working 2× harder

### Scenarios Featured in the RNP

**Scenarios 2 and 3 are the primary examples used in the Render Network Proposal** to demonstrate policy behavior under extreme conditions:

- **Scenario 2 (Market Downturn)** shows how the policy protects node operators during a prolonged bear market by making controlled, gradual decreases and frequently hitting the no-change buffer to prevent excessive volatility.

- **Scenario 3 (Explosive Growth)** shows how the policy responds to rapid network adoption by increasing in controlled steps (capped at 10% per rebalance) to ensure burns exceed issuance while maintaining fair compensation for operators.

### Viewing Results

After running a scenario, results are saved to:
- `reports/future_sims/{scenario_name}.html` - Interactive chart
- `reports/future_sims/{scenario_name}.csv` - Detailed data
- `reports/future_sims/{scenario_name}_decisions.csv` - Decision log

Open the HTML file in your browser to explore the interactive visualization showing:
- How the policy level evolves
- How burns/nodes change over time
- When rebalance decisions occur
- Pay band boundaries (1× to 3× anchor)

### Understanding Scenario Parameters

Each scenario is defined by:
- **`name`**: Descriptive label for the simulation
- **`base`**: What drives the projection
  - `"burn"` - Demand-side (network usage)
  - `"node"` - Supply-side (operator activity)
  - `"hybrid"` - Both factors
- **`curve`**: Growth pattern
  - `"linear"` - Steady change
  - `"exponential"` - Accelerating change
  - `"s-curve"` - Slow start, rapid middle, slow end
- **`periods`**: Number of future epochs (36-52 typical)
- **`multiplier`**: Final value relative to current
  - `1.5` = 50% increase
  - `0.4` = 60% decrease  
  - `3.0` = 200% increase
- **`start_policy`**: Current policy level to start from

For more details on each scenario and their expected behaviors, see the comments in `policySimulation.py`.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or suggestions:

- **Issues:** [GitHub Issues](https://github.com/nickypockets/Render-Forward-Guidance-Tool/issues)
- **Discussions:** [GitHub Discussions](https://github.com/nickypockets/Render-Forward-Guidance-Tool/discussions)

## Acknowledgments

- Render Network for providing public APIs
- Dune Analytics for blockchain data infrastructure
- The Render community for feedback and support
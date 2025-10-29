# Quick Start Guide

Get up and running with the Render Forward Guidance Tool in 15 minutes.

## Prerequisites Check

Before starting, ensure you have:

- [ ] Python 3.10 or higher installed
- [ ] Git installed
- [ ] A Dune Analytics account (free tier is fine)
- [ ] Text editor (VS Code, Notepad++, or any editor)

Check Python version:
```bash
python --version
# Should show Python 3.10.x or higher
```

## Step 1: Get Dune API Key (5 minutes)

1. Go to [dune.com](https://dune.com)
2. Sign up or log in (free account is sufficient)
3. Click your profile → **Settings** → **API**
4. Click **"Create new API key"**
5. Copy the generated key (you'll need it in Step 4)

**Keep this key safe!** You'll paste it into a file in a moment.

## Step 2: Clone Repository (1 minute)

```bash
# Open your terminal (PowerShell on Windows, Terminal on Mac/Linux)
# Navigate to where you want the project
cd Documents  # or wherever you prefer

# Clone the repository
git clone https://github.com/nickypockets/Render-Forward-Guidance-Tool.git

# Enter the project folder
cd Render-Forward-Guidance-Tool
```

## Step 3: Set Up Python Environment (3 minutes)

```bash
# Create a virtual environment (keeps dependencies isolated)
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# You should see (.venv) in your terminal prompt now

# Install all required packages
pip install -r requirements.txt
```

**Wait for installation to complete** (may take 2-3 minutes).

## Step 4: Configure API Key (2 minutes)

Create your `.env` configuration file:

### Windows (PowerShell):
```powershell
# Copy the example file
Copy-Item .env.example .env

# Open .env in Notepad
notepad .env
```

### macOS/Linux:
```bash
# Copy the example file
cp .env.example .env

# Open .env in your default text editor
nano .env
# or: code .env  (if you have VS Code)
```

In the `.env` file, replace `your_api_key_here` with your actual Dune API key from Step 1:

```env
DUNE_API_KEY=abc123xyz789yourActualKeyHere
```

**Save and close the file.**

## Step 5: Download Data (3-5 minutes)

```bash
# This fetches data from Dune Analytics and Render APIs
python -m tools.downloadData
```

You should see output like:
```
Querying Dune for avail (ID: 4833735)...
✓ Downloaded avail data
Querying Dune for work (ID: 3456781)...
✓ Downloaded work data
Fetching burns from URL...
✓ Downloaded burns data
Fetching OBhrs from URL...
✓ Downloaded OBhrs data
```

**If you see errors**, check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

## Step 6: Process Data (2-3 minutes)

```bash
# Process OBhr liability data
python OBhrProcessing.py

# Process node reward data
python -m tools.nodeProcessing

# (Optional) Process burn data with smoothing
python -m tools.burnProcessing
```

You should see:
```
Processing OBhrs data...
✓ Wrote OBhrs_epoch_tier_totals.csv
✓ Generated obhrs_sma_chart.html

Processing node data...
✓ Wrote node_summary.csv
✓ Generated node_sma_chart.html
```

## Step 7: Run Simulation (1 minute)

```bash
# Run the policy simulation
python policySimulation.py
```

You should see:
```
Processing simulation...
✓ Generated policy_simulation.html
✓ Wrote policy_values.csv
```

## Step 8: View Results

Your results are in the `reports/` folder:

```bash
# Windows (open in default browser):
start reports\policy_simulation.html
start reports\obhrs_sma_chart.html
start reports\node_sma_chart.html

# macOS:
open reports/policy_simulation.html
open reports/obhrs_sma_chart.html
open reports/node_sma_chart.html

# Linux:
xdg-open reports/policy_simulation.html
xdg-open reports/obhrs_sma_chart.html
xdg-open reports/node_sma_chart.html
```

**Or** simply navigate to the `reports/` folder in your file explorer and double-click the HTML files.

## What You Just Created

You now have:

✅ **Interactive policy simulation chart** showing:
   - Historical policy values
   - Upper and lower bands
   - Anchor values
   - Rebalance points

✅ **OBhr trends chart** showing:
   - T2 and T3 liability over time
   - Smoothed trends
   - Total network liability

✅ **Node activity chart** showing:
   - Active node counts by tier
   - Growth trends
   - Network expansion

✅ **CSV data files** with all raw calculations for further analysis

## Next Steps

### Update Data Regularly

Run whenever you want fresh data:
```bash
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

python -m tools.downloadData
python OBhrProcessing.py
python -m tools.nodeProcessing
python policySimulation.py
```

### Customize Simulations

Edit `policySimulation.py` to change parameters:

```python
# Find this class at the top of the file
class PolicySimulation:
    START_POLICY = 47435           # Change starting value
    REBALANCE_PERIOD = 12          # Change rebalance frequency (4-24)
    STEP_QUANTUM = 0.05            # Change adjustment step size
    NOCHANGE_BUFFER = 0.10         # Change no-change threshold
```

Then run again:
```bash
python policySimulation.py
```

### Run Future Scenarios

Add this to the bottom of `policySimulation.py` (before running):

```python
if __name__ == "__main__":
    sim = PolicySimulation(skip_node_summary_regen=True)
    sim.run()
    
    # Add these lines for future projections:
    sim.run_scenario(
        name="growth_scenario",
        base="burn",
        curve="exponential",
        periods=52,
        multiplier=15,
        start_policy=15000.0
    )
```

Results will be in `reports/future_sims/growth_scenario_simulation.html`.

### Learn More

- **Full documentation:** [README.md](../README.md)
- **Configuration options:** [docs/CONFIGURATION.md](CONFIGURATION.md)
- **API reference:** [docs/API_REFERENCE.md](API_REFERENCE.md)
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Common First-Time Issues

### "Python not recognized"
- Python not installed or not in PATH
- Solution: Reinstall Python, check "Add to PATH" during installation

### "pip not recognized"
- Pip not installed with Python
- Solution: `python -m ensurepip --upgrade`

### "DUNE_API_KEY not found"
- .env file not created or in wrong location
- Solution: Ensure `.env` is in project root folder

### "Module not found"
- Virtual environment not activated
- Solution: Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)

### "No data downloaded"
- Network issue or invalid API key
- Solution: Check internet connection, verify API key is correct in `.env`

## Getting Help

If you get stuck:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for your specific error
2. Search [GitHub Issues](https://github.com/nickypockets/Render-Forward-Guidance-Tool/issues)
3. Ask in [GitHub Discussions](https://github.com/nickypockets/Render-Forward-Guidance-Tool/discussions)
4. Create a new issue with:
   - Your Python version
   - Your operating system
   - The exact error message
   - What you were trying to do

## Success!

You're now ready to analyze Render Network data and run policy simulations. Enjoy exploring the data!

---

**Pro tip:** Bookmark the `reports/` folder in your browser for quick access to updated charts.

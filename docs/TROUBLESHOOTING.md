# Troubleshooting Guide

Common issues and solutions for the Render Forward Guidance Tool.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Data Download Issues](#data-download-issues)
- [Processing Issues](#processing-issues)
- [Simulation Issues](#simulation-issues)
- [Chart Generation Issues](#chart-generation-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### Python Version Problems

**Problem:** "SyntaxError" or "invalid syntax" errors

**Cause:** Using Python 3.9 or earlier (project requires 3.10+)

**Solution:**
```bash
# Check your Python version
python --version

# Should show Python 3.10 or higher
# If not, install Python 3.11+
```

Install Python 3.11+ from [python.org](https://www.python.org/downloads/) and ensure it's in your PATH.

### Virtual Environment Issues

**Problem:** "Module not found" even after pip install

**Cause:** Not using virtual environment, or wrong environment activated

**Solution:**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# Verify activation (should show .venv path)
which python  # macOS/Linux
where python  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Dependency Installation Fails

**Problem:** "ERROR: Could not build wheels for X"

**Cause:** Missing build tools or incompatible versions

**Solution for Windows:**
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Solution for macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

pip install -r requirements.txt
```

**Solution for Linux:**
```bash
# Install build essentials
sudo apt-get update
sudo apt-get install python3-dev build-essential

pip install -r requirements.txt
```

---

## Configuration Issues

### Missing API Key

**Problem:** "DUNE_API_KEY not found" or "EnvironmentError"

**Cause:** `.env` file missing or incorrectly configured

**Solution:**
```bash
# 1. Create .env file in project root
# Windows PowerShell:
New-Item .env -ItemType File

# macOS/Linux:
touch .env

# 2. Open .env in text editor and add:
DUNE_API_KEY=your_actual_api_key_here

# 3. Verify .env is in correct location
ls .env  # Should show the file

# 4. Restart your script
python -m tools.downloadData
```

**Common mistakes:**
- `.env` in wrong folder (must be project root)
- Extra quotes around API key (don't use quotes)
- Typo in variable name (must be exactly `DUNE_API_KEY`)
- Space before/after equals sign (use `KEY=value`, not `KEY = value`)

### Invalid Dune API Key

**Problem:** "401 Unauthorized" or "403 Forbidden"

**Cause:** API key is invalid or expired

**Solution:**
1. Go to [dune.com/settings/api](https://dune.com/settings/api)
2. Generate a new API key
3. Replace the key in your `.env` file
4. Save and retry

### Missing Configuration Files

**Problem:** "FileNotFoundError: settings/download_ids.json"

**Cause:** Configuration file missing or corrupted

**Solution:**
```bash
# Recreate settings/download_ids.json
# Create settings directory if needed
mkdir settings  # Windows (cmd): md settings

# Create file with this content:
```

```json
{
  "avail": 4833735,
  "work": 3456781,
  "burns": "https://stats.renderfoundation.com/epoch-burn-stats-data.json",
  "OBhrs": "https://stats.renderfoundation.com/liability-epochs-data.json"
}
```

---

## Data Download Issues

### Network Timeout

**Problem:** "TimeoutError" or "Connection timed out"

**Cause:** Slow network or large query

**Solution:**
```python
# Increase timeout in tools/downloadData.py
# Find this line and increase the value:
resp = requests.get(url, timeout=300)  # Increase from 30 to 300
```

### API Rate Limiting

**Problem:** "429 Too Many Requests"

**Cause:** Exceeding Dune API rate limits

**Solution:**
- Wait 1-2 minutes and retry
- Upgrade to Dune Pro for higher limits
- Don't run download script too frequently

### Incomplete Data Downloaded

**Problem:** CSV files are empty or have fewer rows than expected

**Cause:** Query execution failed or incomplete

**Solution:**
```bash
# Check Dune query status manually
# Go to: https://dune.com/queries/4833735 (for avail query)
# Click "Run" to verify query works

# Delete cached data and re-download
rm -rf data/avail_data data/work_data  # macOS/Linux
Remove-Item -Recurse data/avail_data, data/work_data  # Windows

python -m tools.downloadData
```

### JSON Parsing Errors

**Problem:** "JSONDecodeError: Expecting value"

**Cause:** API returned non-JSON response (error page, HTML, etc.)

**Solution:**
```python
# Check if URL is accessible
import requests
resp = requests.get("https://stats.renderfoundation.com/epoch-burn-stats-data.json")
print(resp.status_code)  # Should be 200
print(resp.text[:200])   # Should start with { or [

# If URL changed, update settings/download_ids.json
```

---

## Processing Issues

### Missing Input Files

**Problem:** "FileNotFoundError: data/node_summary.csv"

**Cause:** Processing steps run out of order

**Solution:**
```bash
# Run in correct order:
# 1. Download data
python -m tools.downloadData

# 2. Process nodes (creates node_summary.csv)
python -m tools.nodeProcessing

# 3. Process OBhrs (requires node_summary.csv)
python OBhrProcessing.py

# 4. Run simulation
python policySimulation.py
```

### Data Type Errors

**Problem:** "ValueError: could not convert string to float"

**Cause:** Unexpected data format in CSV/JSON

**Solution:**
```python
# Check your data files for issues
import pandas as pd

# Load and inspect
df = pd.read_csv('data/avail_data/avail.csv')
print(df.info())  # Check column types
print(df.head())  # Check first few rows

# Look for non-numeric values in reward columns
print(df['$RENDER Rewards'].unique())

# Clean data if needed
df['$RENDER Rewards'] = pd.to_numeric(df['$RENDER Rewards'], errors='coerce')
df = df.dropna()
```

### Memory Errors

**Problem:** "MemoryError" or system freezes

**Cause:** Large datasets with insufficient RAM

**Solution:**
```python
# Process data in chunks
import pandas as pd

# Instead of:
df = pd.read_csv('large_file.csv')

# Use:
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    process_chunk(chunk)
```

Or increase Python's available memory:
```bash
# Linux/macOS
ulimit -v 4194304  # 4GB limit

# Or close other applications to free RAM
```

### Tier Classification Issues

**Problem:** Many wallets classified as "unmapped"

**Cause:** Node summary doesn't have tier information for recent wallets

**Solution:**
```bash
# Regenerate node summary with latest data
python -m tools.downloadData  # Get latest rewards
python -m tools.nodeProcessing  # Reclassify wallets
python OBhrProcessing.py  # Reprocess with new classifications
```

---

## Simulation Issues

### Invalid Rebalance Period

**Problem:** "ValueError: rebalance_period must be between 4 and 24"

**Cause:** Configured rebalance period out of valid range

**Solution:**
```python
# Use valid range (4-24 epochs)
sim = PolicySimulation(rebalance_period=12)  # ✓ Valid
# Not: rebalance_period=2  # ✗ Too short
# Not: rebalance_period=30  # ✗ Too long
```

### Simulation Produces NaN Values

**Problem:** Policy values show as `NaN` in output

**Cause:** Missing or invalid input data (OBhrs, rewards, etc.)

**Solution:**
```bash
# Verify all input files exist and have data
ls -lh data/OBhrs_data/OBhrs.json
ls -lh data/work_data/work.csv
ls -lh data/avail_data/avail.csv

# Check files are not empty
wc -l data/work_data/work.csv  # Should show > 1 line

# Redownload if needed
python -m tools.downloadData
```

### Scenario Generation Fails

**Problem:** "KeyError" or "IndexError" when running scenarios

**Cause:** Historical simulation hasn't been run first

**Solution:**
```python
# Always run base simulation first
sim = PolicySimulation()
sim.run()  # ← Must run this first

# Then run scenarios
sim.run_scenario(name="test", base="burn", curve="linear", periods=52, multiplier=2, start_policy=15000)
```

---

## Chart Generation Issues

### No Chart Output

**Problem:** Script completes but no HTML file generated

**Cause:** Error in chart generation that's silently caught

**Solution:**
```python
# Check if plotly is installed
import plotly
print(plotly.__version__)  # Should show version number

# If missing:
pip install plotly

# Check reports folder exists
import os
os.makedirs('reports', exist_ok=True)

# Rerun chart generation
python policySimulation.py
```

### Charts Show Epochs Instead of Dates

**Problem:** X-axis shows epoch numbers when dates expected

**Cause:** `epoch_map.csv` missing or `use_dates` not set

**Solution:**
```bash
# Generate epoch_map.csv
python -m tools.nodeProcessing  # This creates epoch_map.csv

# Verify file exists
ls data/epoch_map.csv

# Use use_dates parameter
from tools.nodeProcessing import plot_sma_by_tier
plot_sma_by_tier(use_dates=True)
```

### Chart Won't Open in Browser

**Problem:** HTML file exists but won't open or display

**Cause:** Browser security restrictions or corrupted file

**Solution:**
```bash
# Try opening in different browser
# Chrome, Firefox, Edge all work well with Plotly

# Check file size (should be > 100KB for typical charts)
ls -lh reports/policy_simulation.html

# If file is tiny (<1KB), regeneration failed
# Check for errors and regenerate
python policySimulation.py 2>&1 | tee output.log
```

### Chart Performance Issues

**Problem:** Charts load slowly or browser freezes

**Cause:** Too many data points

**Solution:**
```python
# Downsample data for large datasets
import pandas as pd
import plotly.graph_objects as go

# Instead of plotting all points:
# fig.add_trace(go.Scatter(x=all_x, y=all_y))

# Sample every Nth point:
n = 10
fig.add_trace(go.Scatter(x=all_x[::n], y=all_y[::n]))
```

---

## Performance Issues

### Slow Data Downloads

**Problem:** Download script takes very long

**Solution:**
```bash
# Check network speed
# Dune queries can take 1-5 minutes for large datasets

# For incremental updates (faster):
# Data is automatically merged, so subsequent downloads only fetch new rows

# For faster repeated runs, use cached data:
# Don't delete data/ folder between runs
```

### Slow Node Processing

**Problem:** `nodeProcessing` takes several minutes

**Solution:**
```python
# Skip node summary regeneration if data hasn't changed
sim = PolicySimulation(skip_node_summary_regen=True)
sim.run()

# This reuses existing node_summary.csv
# Only regenerate when avail.csv or work.csv changes
```

### Slow Simulations

**Problem:** Policy simulation is slow

**Solution:**
```python
# Reduce rebalance frequency (fewer calculations)
sim = PolicySimulation(rebalance_period=24)  # Less frequent rebalancing

# Skip node regeneration
sim = PolicySimulation(skip_node_summary_regen=True)

# Process data once, run multiple scenarios
sim = PolicySimulation(skip_node_summary_regen=True)
sim.run()
sim.run_scenario(...)  # Fast, uses cached processing
sim.run_scenario(...)  # Fast, uses cached processing
```

---

## Path Issues

### Working Directory Problems

**Problem:** "FileNotFoundError" when running from different directories

**Cause:** Old version of code used relative paths

**Solution:**
```bash
# Update to latest version
git pull origin main

# All paths are now file-relative, should work from anywhere

# If still having issues, always run from project root:
cd /path/to/Render-Forward-Guidance-Tool
python policySimulation.py
```

### Permission Errors

**Problem:** "PermissionError: [Errno 13] Permission denied"

**Cause:** Insufficient permissions to write files

**Solution:**
```bash
# Linux/macOS: Check folder permissions
ls -la data/
chmod -R u+w data/  # Give yourself write permission

# Windows: Run as administrator or check folder properties

# Ensure reports folder is writable
mkdir -p reports  # Linux/macOS
md reports  # Windows
```

---

## Getting More Help

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your script
python policySimulation.py
```

### Check Versions

```python
import sys
import pandas as pd
import numpy as np
import plotly

print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Plotly: {plotly.__version__}")
```

Required versions:
- Python: 3.10+
- pandas: 2.0+
- numpy: 1.24+
- plotly: 5.14+

### Still Need Help?

1. **Check existing issues:** [GitHub Issues](https://github.com/nickypockets/Render-Forward-Guidance-Tool/issues)
2. **Search discussions:** [GitHub Discussions](https://github.com/nickypockets/Render-Forward-Guidance-Tool/discussions)
3. **Create new issue:** Include:
   - Python version (`python --version`)
   - Operating system
   - Full error message
   - Steps to reproduce
   - What you've already tried

### Reporting Bugs

When reporting bugs, include:

```bash
# System info
python --version
pip list | grep -E "pandas|numpy|plotly|dune-client"

# Error output
python policySimulation.py 2>&1 | tee error.log

# File sizes (to check if data downloaded correctly)
ls -lh data/**/*
```

Attach `error.log` to your issue report.

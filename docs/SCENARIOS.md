# Future Scenario Projections

The Render Forward Guidance Tool includes 7 pre-configured future scenarios that demonstrate how the policy responds to different market conditions. These scenarios serve as educational examples to help understand policy behavior under various circumstances.

## Table of Contents

- [Overview](#overview)
- [Running Scenarios](#running-scenarios)
- [RNP Featured Scenarios](#rnp-featured-scenarios)
- [All Scenario Descriptions](#all-scenario-descriptions)
  - [Burn-Based Scenarios (Demand-Driven)](#burn-based-scenarios-demand-driven)
  - [Node-Based Scenarios (Supply-Driven)](#node-based-scenarios-supply-driven)
- [Understanding Results](#understanding-results)
- [Custom Scenarios](#custom-scenarios)

---

## Overview

Future scenarios project how the policy level might evolve under different network conditions. Each scenario is designed to illustrate specific aspects of the policy mechanism:

- How the policy responds to growth or decline
- How cap mechanisms prevent excessive volatility
- How node activity influences policy decisions
- How tier composition affects the anchor calculation

**Two scenarios (Market Downturn and Explosive Growth) are featured in the Render Network Proposal (RNP)** as primary examples demonstrating policy behavior under extreme conditions.

---

## Running Scenarios

All scenarios are pre-configured at the end of `policySimulation.py`. To run any scenario:

### Method 1: Uncomment in Script

1. Open `policySimulation.py`
2. Scroll to the bottom (after `if __name__ == "__main__":`)
3. Find the scenario you want (e.g., "Market Downturn")
4. Uncomment the `sim.run_scenario()` call
5. Run: `python policySimulation.py`

```python
# Example: Run Market Downturn scenario
sim.run_scenario(
    name="market_downturn", 
    base="burn", 
    curve="linear", 
    periods=52, 
    multiplier=0.4, 
    start_policy=15000.0
)
```

### Method 2: Run from Python

```python
from policySimulation import PolicySimulation

# Initialize (skip_node_summary_regen=True for speed)
sim = PolicySimulation(skip_node_summary_regen=True)

# Run any scenario
sim.run_scenario(
    name="market_downturn",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=0.4,
    start_policy=15000.0
)
```

### Output Location

Results are saved to `reports/future_sims/`:
- `{scenario_name}.html` - Interactive visualization
- `{scenario_name}.csv` - Detailed epoch-by-epoch data
- `{scenario_name}_decisions.csv` - Decision log with reasons

---

## RNP Featured Scenarios

Two scenarios are **prominently featured in the Render Network Proposal (RNP)** to demonstrate policy behavior under extreme market conditions:

### ⭐ Scenario 2: Market Downturn (Bear Market)

**Featured in RNP to demonstrate downside protection**

```python
sim.run_scenario(
    name="market_downturn",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=0.4,
    start_policy=15000.0
)
```

**What it shows:**
- Burns decline 60% over 1 year (52 epochs)
- Policy makes controlled, gradual decreases
- No-change buffer activates frequently to prevent excessive volatility
- Demonstrates how the policy protects node operators during prolonged downturns

**Expected behavior:**
- Policy decreases in steps (typically -10% per rebalance when needed)
- Buffer prevents changes when anchor is within ±10% of current policy
- Helps maintain operator sustainability during difficult market periods

**Key insights from RNP:**
This scenario validates that the policy mechanism provides adequate protection for node operators while still being responsive to genuine market changes. The no-change buffer is particularly important here, preventing policy thrashing during volatile periods.

---

### ⭐ Scenario 3: Explosive Growth (Bull Market)

**Featured in RNP to demonstrate upside responsiveness**

```python
sim.run_scenario(
    name="explosive_growth",
    base="burn",
    curve="exponential",
    periods=36,
    multiplier=3.0,
    start_policy=15000.0
)
```

**What it shows:**
- Burns increase 200% (3× current) over 9 months (36 epochs)
- Exponential curve simulates accelerating adoption
- Policy increases aggressively but in controlled steps
- Cap escalation mechanism allows faster response to sustained growth

**Expected behavior:**
- Policy increases at maximum rate (10% per rebalance)
- Cap escalation: consecutive increases can compound to 20%+
- Ensures burns exceed issuance while maintaining fair operator compensation
- Demonstrates the policy can keep pace with rapid growth

**Key insights from RNP:**
This scenario demonstrates that even during viral adoption or major bull runs, the policy mechanism maintains stability while ensuring the network remains deflationary. The 10% cap prevents single-epoch shocks, while cap escalation allows the policy to track sustained growth trends.

---

## All Scenario Descriptions

### Burn-Based Scenarios (Demand-Driven)

These scenarios focus on changes in network usage and token burns (demand side).

#### 1. Steady Growth

**Scenario:** Healthy, sustainable network growth

```python
sim.run_scenario(
    name="steady_growth",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=1.5,
    start_policy=15000.0
)
```

**Parameters:**
- Burns increase 50% over 1 year (52 epochs)
- Linear curve (steady, predictable growth)
- Start policy: 15,000 tokens

**What it demonstrates:**
- How policy responds to gradual, predictable growth
- The 10% cap preventing rapid jumps
- Smooth policy increases tracking burn trends

**Expected behavior:**
- Policy increases gradually but steadily
- Increases typically capped at 10% per rebalance
- Policy stays within pay bands (1× to 3× anchor)

**Use case:** Understanding baseline policy behavior during healthy network growth.

---

#### 2. Market Downturn ⭐

**Scenario:** Bear market / recession / prolonged decline

*See [RNP Featured Scenarios](#rnp-featured-scenarios) section above for full details.*

---

#### 3. Explosive Growth ⭐

**Scenario:** Viral adoption / major bull run / rapid acceleration

*See [RNP Featured Scenarios](#rnp-featured-scenarios) section above for full details.*

---

### Node-Based Scenarios (Supply-Driven)

These scenarios focus on changes in node operator participation and activity (supply side).

#### 4. Node Expansion

**Scenario:** Organic growth in node operator base

```python
sim.run_scenario(
    name="node_expansion",
    base="node",
    curve="linear",
    periods=52,
    multiplier=1.5,
    start_policy=15000.0
)
```

**Parameters:**
- Node count/activity increases 50% over 1 year (52 epochs)
- Linear curve (steady operator growth)
- Start policy: 15,000 tokens

**What it demonstrates:**
- How policy responds to growing supply capacity
- The node multiplier component in action
- Supply-side influence on anchor calculation

**Expected behavior:**
- Policy increases as node capacity grows
- Node multiplier pushes anchor higher
- Policy adjusts to maintain fair compensation

**Use case:** Understanding how operator expansion affects policy recommendations.

---

#### 5. Node Attrition

**Scenario:** Operator exodus / difficult period for nodes

```python
sim.run_scenario(
    name="node_attrition",
    base="node",
    curve="linear",
    periods=52,
    multiplier=0.65,
    start_policy=15000.0
)
```

**Parameters:**
- Node count/activity declines 35% over 1 year (52 epochs)
- Linear curve (steady attrition)
- Start policy: 15,000 tokens

**What it demonstrates:**
- How policy protects against capacity loss
- Adaptive response to supply-side constraints
- Preventing network capacity crisis

**Expected behavior:**
- Policy decreases to match reduced capacity
- Node multiplier lowers anchor value
- Helps retain remaining operators

**Use case:** Understanding policy response when operators leave the network.

---

#### 6. Tier Migration

**Scenario:** Quality upgrade - operators shift from T2 to T3

```python
sim_tier = PolicySimulation(skip_node_summary_regen=True)
df, steps, meta = sim_tier.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier={"T2": 0.7, "T3": 1.4}
)
```

**Parameters:**
- T2 nodes decline 30%, T3 nodes grow 40%
- Simulates migration over 10 months (40 epochs from historical endpoint)
- Uses `run_hypothetical()` for tier-specific control

**What it demonstrates:**
- How policy handles compositional changes
- Equal-pain factor adjusting to tier mix
- Dynamic tier weights (f2, f3) in anchor calculation

**Expected behavior:**
- Policy adjusts based on changing tier mix
- Equal-pain factor recalibrates
- Higher T3 proportion typically increases anchor (T3 nodes have higher costs)

**Use case:** Understanding how network composition affects policy decisions.

**Note:** This scenario uses `run_hypothetical()` instead of `run_scenario()` because it requires tier-specific multipliers. See the commented example in `policySimulation.py`.

---

#### 7. Activity Surge

**Scenario:** Existing nodes work harder / increased utilization

```python
sim.run_scenario(
    name="activity_surge",
    base="node",
    curve="exponential",
    periods=36,
    multiplier=2.0,
    start_policy=15000.0
)
```

**Parameters:**
- Node activity (hours worked) doubles over 9 months (36 epochs)
- Exponential curve (accelerating utilization)
- Start policy: 15,000 tokens

**What it demonstrates:**
- How increased utilization affects the anchor
- Hours-change multiplier component (x2, x3)
- Supply-side activity influence separate from node count

**Expected behavior:**
- Policy increases as nodes work harder
- Hours multiplier component pushes anchor higher
- Rewards increased operator commitment

**Use case:** Understanding how operator activity levels affect policy independently of node count changes.

---

## Understanding Results

### Interactive Charts

Each scenario generates an interactive HTML chart showing:

- **Policy Line (green):** Step function showing policy changes
- **1× Anchor (red):** Lower pay band boundary
- **3× Anchor (blue):** Upper pay band boundary
- **Shaded Area:** Pay band range (1× to 3× anchor)
- **Diamond Markers:** Rebalance epochs where decisions occur

**Interactive features:**
- Hover over points to see exact values
- Zoom into specific time periods
- Click legend items to toggle traces on/off
- Download as PNG image

### CSV Data Files

**{scenario_name}.csv** contains epoch-by-epoch data:
- Epoch number
- Burn tokens and USD values
- SMA (smoothed moving average) of burns
- Growth rates
- Node counts by tier (T2, T3)
- Hours worked by tier
- Anchor values
- Policy level

**{scenario_name}_decisions.csv** contains decision logic:
- Epoch where decision occurred
- Previous policy value
- New policy value
- Decision type (change, no_change, hold)
- Reason code
- Cap used
- Step percentage
- Full explanation text

### Key Metrics to Watch

1. **Anchor Value:** Target policy based on burns and node activity
2. **Policy vs. Anchor:** How close policy stays to anchor
3. **Rebalance Frequency:** How often decisions trigger vs. hit buffer
4. **Growth Rates:** Burn growth and node/hour change percentages
5. **Cap Usage:** When and how often 10% cap is hit

---

## Custom Scenarios

You can create your own scenarios by adjusting parameters:

### Basic Custom Scenario

```python
sim.run_scenario(
    name="my_custom_scenario",
    base="burn",           # or "node" or "hybrid"
    curve="linear",        # or "exponential" or "s-curve"
    periods=40,            # number of future epochs
    multiplier=1.2,        # 20% increase
    start_policy=15000.0   # current policy level
)
```

### Parameter Guide

**base:** What drives the projection
- `"burn"`: Varies burn rates (demand)
- `"node"`: Varies node activity (supply)
- `"hybrid"`: Varies both (50/50 split of effect)

**curve:** Growth pattern
- `"linear"`: Steady change (constant rate)
- `"exponential"`: Accelerating change (growth compounds)
- `"s-curve"`: Slow start, rapid middle, slow end (sigmoid)
- `"bezier"`: Custom schedule (requires multiplier as list of tuples)

**periods:** How far to project
- Typical: 36-52 epochs (9 months to 1 year)
- Longer projections become more speculative

**multiplier:** Final value relative to current
- `> 1.0`: Growth scenarios
  - `1.2` = 20% increase
  - `1.5` = 50% increase
  - `2.0` = 100% increase (double)
- `< 1.0`: Decline scenarios
  - `0.9` = 10% decrease
  - `0.6` = 40% decrease
  - `0.5` = 50% decrease (half)
- For bezier curve: list of `(multiplier, duration)` tuples

**start_policy:** Current policy to start from
- Use the actual current policy level
- Or hypothetical value to test different starting points

### Advanced: Custom Cadence

Change the rebalancing frequency:

```python
# 8-epoch cadence instead of default 12
sim_custom = PolicySimulation(
    rebalance_period=8,
    first_rebalance_epoch=109,  # when to start in absolute epochs
    skip_node_summary_regen=True
)

sim_custom.run_scenario(
    name="steady_growth_8ep",
    base="burn",
    curve="linear",
    periods=52,
    multiplier=1.5,
    start_policy=15000.0
)
```

### Advanced: Tier-Specific Changes

Use `run_hypothetical()` for tier-specific multipliers:

```python
df, steps, meta = sim.run_hypothetical(
    start_epoch=97,
    start_policy=15000.0,
    nodes_multiplier={"T2": 1.2, "T3": 0.8},  # T2 up 20%, T3 down 20%
    hours_multiplier={"T2": 1.5, "T3": 1.5}   # Both tiers work 50% more
)
```

---

## Best Practices

### When to Use Each Scenario Type

**Burn-Based (Demand):**
- Testing market sensitivity
- Understanding deflationary mechanisms
- Modeling adoption curves
- Revenue/usage projections

**Node-Based (Supply):**
- Capacity planning
- Operator incentive analysis
- Network resilience testing
- Compensation fairness validation

**Hybrid:**
- Comprehensive stress testing
- Balanced growth modeling
- Complex market scenarios

### Interpreting Results

**Good signs:**
- Policy stays within pay bands most of the time
- Changes are gradual and predictable
- Buffer prevents excessive volatility
- Anchor tracks underlying trends

**Warning signs:**
- Policy hits upper/lower band frequently (may need band adjustment)
- Frequent oscillation (may need larger buffer)
- Policy can't keep pace with rapid changes (may need cap escalation)

### Scenario Limitations

**Remember:**
1. **Projections are not predictions** - they show "what if" scenarios, not forecasts
2. **Simplified assumptions** - real networks have more complexity
3. **Historical data limitations** - past patterns may not continue
4. **External factors ignored** - market conditions, competition, technology changes

Use scenarios to understand mechanism behavior, not to predict specific future values.

---

## Questions?

For more information:
- [API Reference](API_REFERENCE.md) - Complete function documentation
- [README](../README.md) - General usage guide
- [GitHub Issues](https://github.com/nickypockets/Render-Forward-Guidance-Tool/issues) - Report problems or ask questions

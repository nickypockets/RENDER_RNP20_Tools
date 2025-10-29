"""Policy simulation module refactored into an OOP-style container.

The original procedural logic has been preserved but reorganized into a
single PolicySimulation class. This keeps the data flow self-contained and
makes the processing easier to test and reuse.
"""

from pathlib import Path
import json
from typing import Optional, Tuple


import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PolicySimulation:
    """Container for policy simulation configuration and execution.

    Usage:
      sim = PolicySimulation()
      sim.run()
    """

    # defaults (kept as instance attributes to make them tunable)
    START_POLICY = 47435
    REBALANCE_PERIOD = 12
    FIRST_REBALANCE_EPOCH = 12
    F2, F3 = 2 / 3, 1 / 3
    LAMBDA2 = 1.0
    LAMBDA3 = 1.0
    BAND_MULTIPLIER_UPPER = 3.0
    BAND_MULTIPLIER_LOWER = 1.0
    TRIGGER_ABOVE_UPPER = 1
    NOCHANGE_BUFFER = 0.10
    STEP_QUANTUM = 0.05
    ROUND_TO = 100
    DEFAULT_EURUSD = 1.0

    def __init__(
        self,
        burns_json: Optional[Path] = None,
        jobs_csv: Optional[Path] = None,
        avail_csv: Optional[Path] = None,
        tiers_csv: Optional[Path] = None,
        out_html: Optional[Path] = None,
        out_csv: Optional[Path] = None,
        out_classified: Optional[Path] = None,
        rebalance_period: Optional[int] = None,
        first_rebalance_epoch: Optional[int] = None,
        skip_node_summary_regen: bool = False,
    ) -> None:
        # Get project root relative to this file
        project_root = Path(__file__).resolve().parent
        
        # sensible repo-relative defaults (adapted to this project layout)
        self.burns_json = Path(burns_json) if burns_json else project_root / "data" / "OBhrs_data" / "OBhrs.json"
        self.jobs_csv = Path(jobs_csv) if jobs_csv else project_root / "data" / "work_data" / "work.csv"
        self.avail_csv = Path(avail_csv) if avail_csv else project_root / "data" / "avail_data" / "avail.csv"
        self.tiers_csv = Path(tiers_csv) if tiers_csv else project_root / "data" / "node_summary.csv"

        self.out_html = Path(out_html) if out_html else project_root / "reports" / "policy_simulation.html"
        self.out_csv = Path(out_csv) if out_csv else project_root / "reports" / "policy_values.csv"
        self.out_classified = Path(out_classified) if out_classified else project_root / "reports" / "tiered_wallets_classified.csv"

        # Allow instance-level override of rebalance cadence (epochs per rebalance).
        # Acceptable range is 4..24 (inclusive) to keep cadences sensible.
        if rebalance_period is not None:
            try:
                rp = int(rebalance_period)
            except Exception:
                raise ValueError("rebalance_period must be an integer between 4 and 24")
            if rp < 4 or rp > 24:
                raise ValueError("rebalance_period must be between 4 and 24 epochs")
            self.REBALANCE_PERIOD = rp
        else:
            # keep class default
            self.REBALANCE_PERIOD = PolicySimulation.REBALANCE_PERIOD

        # FIRST_REBALANCE_EPOCH defaults to the rebalance period unless explicitly provided
        if first_rebalance_epoch is not None:
            try:
                self.FIRST_REBALANCE_EPOCH = int(first_rebalance_epoch)
            except Exception:
                raise ValueError("first_rebalance_epoch must be an integer")
        else:
            self.FIRST_REBALANCE_EPOCH = int(self.REBALANCE_PERIOD)

        # Cache for expensive operations to avoid redundant data loading
        self._node_summary_cache: Optional[pd.DataFrame] = None
        self._cache_dirty = True
        self.skip_node_summary_regen = skip_node_summary_regen

    # -------------------
    # Data loading helpers
    # -------------------
    @staticmethod
    def _coalesce(df: pd.DataFrame, candidates, default=None):
        for c in candidates:
            if c in df.columns:
                return c
        return default

    def _get_node_summary(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get cached node summary or load it once.

        This method caches the result of aggregate_node_rewards() to avoid
        redundant CSV reading and processing when multiple methods need the
        same data.

        Parameters
        ----------
        force_refresh : bool, optional
            If True, bypass cache and reload data from source.

        Returns
        -------
        pd.DataFrame or None
            The node summary DataFrame, or None if unavailable.
        """
        if not force_refresh and self._node_summary_cache is not None and not self._cache_dirty:
            return self._node_summary_cache

        # If skip_node_summary_regen is True, try to load from CSV instead of regenerating
        if self.skip_node_summary_regen:
            try:
                if self.tiers_csv.exists():
                    ns = pd.read_csv(self.tiers_csv)
                    if ns is not None and not ns.empty:
                        self._node_summary_cache = ns
                        self._cache_dirty = False
                        return ns
            except Exception:
                pass

        try:
            import tools.nodeProcessing as npg
            ns = npg.aggregate_node_rewards()
            if ns is not None and not ns.empty:
                self._node_summary_cache = ns
                self._cache_dirty = False
                return ns
        except Exception:
            pass

        self._node_summary_cache = None
        self._cache_dirty = False
        return None

    def invalidate_cache(self) -> None:
        """Mark cached data as dirty to force reload on next access."""
        self._cache_dirty = True

    def preload_data(self) -> None:
        """Preload and cache node summary data.

        Call this method early (e.g., at the start of run()) to ensure all
        subsequent data loading operations use cached data instead of
        repeatedly calling aggregate_node_rewards().

        This is optional but recommended for workflows that call multiple
        methods (run, explain_latest_policy, scenarios, etc.) on the same
        instance.
        """
        self._get_node_summary()

    def load_burns(self, burns_json: Path) -> pd.DataFrame:
        # Prefer to source burns from tools.burnProcessing if available. That
        # module already normalizes the canonical burns JSON into `df_burns`.
        try:
            import tools.burnProcessing as bp
            # bp.df_burns is expected to be present and contain 'Burns' and optionally 'USDBurn'
            df = getattr(bp, "df_burns", None)
            if df is None or df.empty:
                raise RuntimeError("tools.burnProcessing.df_burns missing or empty")

            burns = df.copy()
            # determine epoch numbering: use an 'id' column if available, otherwise ordinal index
            if "id" in burns.columns:
                burns = burns.rename(columns={"id": "epoch"})
            else:
                burns = burns.reset_index(drop=True)
                burns["epoch"] = burns.index + 1

            # token burns
            if "Burns" in burns.columns:
                burns = burns.rename(columns={"Burns": "burn_tokens"})
            elif "burnedRender" in burns.columns:
                burns = burns.rename(columns={"burnedRender": "burn_tokens"})

            # USD burns
            if "USDBurn" in burns.columns:
                burns = burns.rename(columns={"USDBurn": "burn_usd"})
            elif "burnedRenderUSDCAmt" in burns.columns:
                burns = burns.rename(columns={"burnedRenderUSDCAmt": "burn_usd"})

            # ensure required columns
            if "epoch" not in burns.columns or "burn_tokens" not in burns.columns:
                raise RuntimeError("tools.burnProcessing produced DataFrame without epoch/burn_tokens")

            burns = burns[["epoch", "burn_tokens"] + (["burn_usd"] if "burn_usd" in burns.columns else [])]
            burns = burns.sort_values("epoch").reset_index(drop=True)
            return burns
        except Exception:
            # Fallback: attempt to read raw JSON and map flexible fields (original behavior)
            with open(burns_json, "r") as f:
                data = json.load(f)
            burns = pd.DataFrame(data)
            epoch_col = self._coalesce(burns, ["epoch", "Epoch", "ep"])
            token_col = self._coalesce(burns, ["burn_tokens", "burnTokens", "token_burn", "tokens", "Burns", "burnedRender"])
            usd_col = self._coalesce(burns, ["burn_usd", "burnUSD", "usd_burn", "usd", "USDBurn", "burnedRenderUSDCAmt"])
            if epoch_col is None or token_col is None:
                raise ValueError("burns.json missing required columns (epoch, token burn).")
            burns = burns.rename(columns={epoch_col: "epoch", token_col: "burn_tokens"})
            if usd_col is not None:
                burns = burns.rename(columns={usd_col: "burn_usd"})
            burns = burns.sort_values("epoch").reset_index(drop=True)
            return burns

    def load_nodes_from_tools(self) -> pd.DataFrame:
        """Use tools.nodeProcessing.aggregate_node_rewards to build per-epoch tier shares and node counts.

        Returns a DataFrame with columns: epoch, share_T2, share_T3, nodes_T2, nodes_T3
        """
        try:
            # Use cached node summary to avoid redundant processing
            ns = self._get_node_summary()
            if ns is None or ns.empty:
                raise RuntimeError("nodeProcessing returned empty node summary")

            # detect work epoch columns: workE{n}
            work_cols = [c for c in ns.columns if c.startswith("workE")]
            # detect tier columns tierE{n}
            tier_cols = [c for c in ns.columns if c.startswith("tierE")]

            # determine epoch numbers from work_cols
            epochs = sorted({int(c.replace("workE", "")) for c in work_cols}) if work_cols else []

            rows = []
            for ep in epochs:
                wcol = f"workE{ep}"
                tcol = f"tierE{ep}"
                # total work tokens that epoch
                total = ns[wcol].astype(float).sum()
                # work by tier
                if tcol in ns.columns:
                    t2_mask = ns[tcol] == 'T2'
                    t3_mask = ns[tcol] == 'T3'
                    work_t2 = ns.loc[t2_mask, wcol].astype(float).sum()
                    work_t3 = ns.loc[t3_mask, wcol].astype(float).sum()
                    nodes_t2 = int(((ns[wcol].astype(float) > 0) & (ns[tcol] == 'T2')).sum())
                    nodes_t3 = int(((ns[wcol].astype(float) > 0) & (ns[tcol] == 'T3')).sum())
                else:
                    # if no tier column, approximate all as T3
                    work_t2 = 0.0
                    work_t3 = total
                    nodes_t2 = 0
                    nodes_t3 = int((ns[wcol].astype(float) > 0).sum())

                share_t2 = float(work_t2 / total) if total > 0 else 0.0
                share_t3 = float(work_t3 / total) if total > 0 else 0.0

                rows.append({"epoch": ep, "share_T2": share_t2, "share_T3": share_t3, "nodes_T2": nodes_t2, "nodes_T3": nodes_t3})

            tier_stats = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
            return tier_stats
        except Exception:
            raise

    def load_rewards(self, jobs_csv: Path, avail_csv: Path) -> pd.DataFrame:
        """Return a canonical long-form rewards DataFrame (epoch,wallet,amount,type).

        Strategy:
          1. Prefer `tools.nodeProcessing.aggregate_node_rewards()` output and unpivot it.
          2. Accept pandas.DataFrame inputs for jobs_csv/avail_csv (useful when callers pass in-memory data).
          3. Fall back to reading CSVs from disk if present.
          4. If nothing available, raise an explanatory error.
        """
        # 1) Try nodeProcessing pivot -> long-form conversion (use cached summary)
        try:
            ns = self._get_node_summary()
            if ns is not None and not ns.empty:
                work_cols = [c for c in ns.columns if c.startswith("workE")]
                avail_cols = [c for c in ns.columns if c.startswith("availE")]
                rows = []
                for _, r in ns.iterrows():
                    # Handle both Series.get() and direct attribute access
                    wallet = r.get("recipient_wallet", None) if hasattr(r, 'get') else r["recipient_wallet"]
                    if wallet is None or pd.isna(wallet):
                        continue
                    wallet = str(wallet).lower()
                    for wc in work_cols:
                        try:
                            ep = int(wc.replace("workE", ""))
                        except Exception:
                            continue
                        amt = r.get(wc, 0.0) if hasattr(r, 'get') else r[wc]
                        amt = float(amt or 0.0)
                        if amt and amt != 0.0:
                            rows.append({"epoch": ep, "wallet": wallet, "amount": amt, "type": "work"})
                    for ac in avail_cols:
                        try:
                            ep = int(ac.replace("availE", ""))
                        except Exception:
                            continue
                        amt = r.get(ac, 0.0) if hasattr(r, 'get') else r[ac]
                        amt = float(amt or 0.0)
                        if amt and amt != 0.0:
                            rows.append({"epoch": ep, "wallet": wallet, "amount": amt, "type": "availability"})
                rewards = pd.DataFrame(rows)
                if not rewards.empty:
                    rewards = rewards.groupby(["epoch", "wallet", "type"], as_index=False)["amount"].sum()
                    return rewards
        except Exception as e:
            # nodeProcessing not available or produced no rows; continue to other sources
            print(f"Warning: Could not load from node_summary: {e}")
            pass

        # 2) Accept DataFrame inputs directly
        jobs_df = jobs_csv if isinstance(jobs_csv, pd.DataFrame) else None
        avail_df = avail_csv if isinstance(avail_csv, pd.DataFrame) else None

        # 3) If DataFrames not provided, try reading CSVs if files exist
        if jobs_df is None and Path(jobs_csv).exists():
            jobs_df = pd.read_csv(jobs_csv)
        if avail_df is None and Path(avail_csv).exists():
            avail_df = pd.read_csv(avail_csv)

        if (jobs_df is None or jobs_df.empty) and (avail_df is None or avail_df.empty):
            raise ValueError("No rewards available: node summary empty and no CSV/DataFrame inputs provided.")

        parts = []
        if jobs_df is not None and not jobs_df.empty:
            e_j = self._coalesce(jobs_df, ["epoch", "Epoch", "ep"])
            w_j = self._coalesce(jobs_df, ["wallet", "Wallet", "to", "address", "addr"])
            a_j = self._coalesce(jobs_df, ["amount", "value", "payout", "payout_amount", "render", "render_amount", "RENDER"])
            if e_j is None or w_j is None or a_j is None:
                raise ValueError("Jobs data missing epoch/wallet/amount columns.")
            jobs = jobs_df.rename(columns={e_j: "epoch", w_j: "wallet", a_j: "amount"}).copy()
            jobs["type"] = "work"
            jobs["epoch"] = pd.to_numeric(jobs["epoch"], errors="coerce")
            jobs["amount"] = pd.to_numeric(jobs["amount"], errors="coerce").fillna(0.0)
            parts.append(jobs)

        if avail_df is not None and not avail_df.empty:
            e_a = self._coalesce(avail_df, ["epoch", "Epoch", "ep"])
            w_a = self._coalesce(avail_df, ["wallet", "Wallet", "to", "address", "addr"])
            a_a = self._coalesce(avail_df, ["amount", "value", "payout", "payout_amount", "render", "render_amount", "RENDER"])
            if e_a is None or w_a is None or a_a is None:
                raise ValueError("Availability data missing epoch/wallet/amount columns.")
            av = avail_df.rename(columns={e_a: "epoch", w_a: "wallet", a_a: "amount"}).copy()
            av["type"] = "availability"
            av["epoch"] = pd.to_numeric(av["epoch"], errors="coerce")
            av["amount"] = pd.to_numeric(av["amount"], errors="coerce").fillna(0.0)
            parts.append(av)

        rewards = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        rewards = rewards.dropna(subset=["epoch", "wallet"]) if not rewards.empty else rewards
        if not rewards.empty:
            rewards["wallet"] = rewards["wallet"].astype(str).str.lower()
            rewards = rewards.groupby(["epoch", "wallet", "type"], as_index=False)["amount"].sum()
        return rewards

    def load_tiers(self, tiers_csv: Path) -> pd.DataFrame:
        # Prefer to derive tiers from the node summary produced by tools.nodeProcessing
        try:
            # Use cached node summary to avoid redundant processing
            ns = self._get_node_summary()

            if ns is not None and not ns.empty:
                # find per-epoch tier columns like tierE{n}
                tier_cols = [c for c in ns.columns if c.startswith("tierE")]
                tier_cols_sorted = sorted(tier_cols, key=lambda s: int(s.replace("tierE", "")) if s.replace("tierE", "").isdigit() else 0)
                rows = []
                for _, r in ns.iterrows():
                    wallet = r.get("recipient_wallet") if "recipient_wallet" in r.index else r.get("recipient_wallet")
                    wallet = str(wallet).lower()
                    chosen = None
                    # pick the most recent non-null tier value
                    for col in reversed(tier_cols_sorted):
                        try:
                            val = r.get(col)
                        except Exception:
                            val = None
                        if pd.notna(val) and str(val).strip() != "":
                            chosen = str(val).upper().replace(" ", "")
                            break
                    if chosen in ("T2", "T3"):
                        rows.append({"wallet": wallet, "tier": chosen})

                tiers_df = pd.DataFrame(rows)
                if not tiers_df.empty:
                    tiers_df = tiers_df.drop_duplicates(subset=["wallet"]).reset_index(drop=True)
                    return tiers_df
        except Exception:
            # ignore and fall back to CSV
            pass

        # Fallback: read provided CSV (legacy behavior)
        if not Path(tiers_csv).exists():
            # return empty frame if no source available
            return pd.DataFrame(columns=["wallet", "tier"])

        tiers = pd.read_csv(tiers_csv)
        w = self._coalesce(tiers, ["wallet", "Wallet", "address", "addr", "to"])
        t = self._coalesce(tiers, ["tier", "Tier", "TIER", "node_tier", "nodeTier"])
        if w is None or t is None:
            # return empty instead of raising to allow downstream classification
            return pd.DataFrame(columns=["wallet", "tier"])
        tiers = tiers.rename(columns={w: "wallet", t: "tier"}).copy()
        tiers["wallet"] = tiers["wallet"].astype(str).str.lower()
        tiers["tier"] = tiers["tier"].astype(str).str.upper().str.replace(" ", "")
        tiers["tier"] = tiers["tier"].replace({"2": "T2", "3": "T3", "TIER2": "T2", "TIER3": "T3"})
        tiers = tiers[tiers["tier"].isin(["T2", "T3"])].drop_duplicates(subset=["wallet"]).reset_index(drop=True)
        return tiers

    def _locate_obhrs_epoch_totals(self) -> Optional[Path]:
        """Return the first existing OBhrs epoch totals CSV path, if available."""
        project_root = Path(__file__).resolve().parent

        candidates = [
            project_root / "data" / "OBhrs_data" / "OBhrs_epoch_tier_totals.csv",
            project_root / "data" / "OBhrs_epoch_tier_totals.csv",
            project_root / "reports" / "OBhrs_epoch_tier_totals.csv",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    # -------------------
    # Processing helpers
    # -------------------
    def classify_new_wallets(self, rewards: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
        known = set(tiers["wallet"]) if not tiers.empty else set()
        last_epochs = rewards["epoch"].max()
        window = sorted(rewards["epoch"].dropna().unique())[-2:] if pd.notnull(last_epochs) else []
        recent = rewards[rewards["epoch"].isin(window)].copy()

        pivot = recent.pivot_table(index="wallet", columns="type", values="amount", aggfunc="sum", fill_value=0.0)
        pivot = pivot.rename_axis(None, axis=1).reset_index()
        pivot = pivot[~pivot["wallet"].isin(known)].copy()
        pivot["ratio_wa"] = np.where(pivot.get("availability", 0) > 0, pivot.get("work", 0) / pivot.get("availability", 0), np.inf)
        pivot["tier"] = np.where(pivot["ratio_wa"] >= 4.0, "T2", "T3")
        pivot["newer_node"] = True

        newer = pivot[["wallet", "tier", "newer_node"]].copy()
        tiers2 = tiers.copy()
        tiers2["newer_node"] = False
        tiers_all = pd.concat([tiers2, newer], ignore_index=True)
        tiers_all = tiers_all.drop_duplicates(subset=["wallet"], keep="first")
        return tiers_all

    def per_epoch_tier_shares(self, rewards: pd.DataFrame, tiers_all: pd.DataFrame) -> pd.DataFrame:
        rw = rewards.merge(tiers_all[["wallet", "tier"]], how="left", on="wallet")
        presence = rw.groupby(["epoch", "tier", "wallet"], as_index=False)["amount"].sum()
        nodes = presence.groupby(["epoch", "tier"], as_index=False)["wallet"].nunique().rename(columns={"wallet": "nodes"})

        work = rw[rw["type"] == "work"].copy()
        by_tier = work.groupby(["epoch", "tier"], as_index=False)["amount"].sum().rename(columns={"amount": "work_tokens"})
        total_work = by_tier.groupby("epoch", as_index=False)["work_tokens"].sum().rename(columns={"work_tokens": "work_total"})
        shares = by_tier.merge(total_work, on="epoch", how="left")
        shares["share"] = np.where(shares["work_total"] > 0, shares["work_tokens"] / shares["work_total"], 0.0)

        shares_p = shares.pivot_table(index="epoch", columns="tier", values="share", aggfunc="first").reset_index()
        shares_p.columns.name = None
        for c in ["T2", "T3"]:
            if c not in shares_p.columns:
                shares_p[c] = 0.0
        shares_p = shares_p.rename(columns={"T2": "share_T2", "T3": "share_T3"})

        nodes_p = nodes.pivot_table(index="epoch", columns="tier", values="nodes", aggfunc="first").reset_index()
        nodes_p.columns.name = None
        for c in ["T2", "T3"]:
            if c not in nodes_p.columns:
                nodes_p[c] = 0
        nodes_p = nodes_p.rename(columns={"T2": "nodes_T2", "T3": "nodes_T3"})

        tier_stats = shares_p.merge(nodes_p, on="epoch", how="outer").sort_values("epoch").reset_index(drop=True)
        return tier_stats

    def infer_hours_per_tier(self, burns_usd: pd.Series, share_T2: pd.Series, share_T3: pd.Series, eurusd: float = None):
        eurusd = eurusd if eurusd is not None else self.DEFAULT_EURUSD
        eur_T2 = (burns_usd.fillna(0.0) * share_T2.fillna(0.0)) / max(eurusd, 1e-9)
        eur_T3 = (burns_usd.fillna(0.0) * share_T3.fillna(0.0)) / max(eurusd, 1e-9)
        hours_T2 = eur_T2 * 100.0
        hours_T3 = eur_T3 * 200.0
        return hours_T2, hours_T3

    # math helpers (kept as instance methods so constants are easily accessible)
    def node_change_multiplier(self, f2, f3, r2, r3, x2, x3):
        # Handle NaN/inf inputs by returning 1.0 (no change)
        try:
            vals = [f2, f3, r2, r3, x2, x3]
            if any(not np.isfinite(v) for v in vals):
                return 1.0
        except (TypeError, ValueError):
            return 1.0
        num = f2 * (1.0 + r2) * (1.0 + x2) + f3 * (1.0 + r3) * (1.0 + x3)
        den = f2 * (1.0 + x2) + f3 * (1.0 + x3)
        result = num / max(den, 1e-12)
        return result if np.isfinite(result) else 1.0

    def min_max_equal_pain(self, f2, f3, x2, x3, lam2, lam3, r2=0.0, r3=0.0):
        # Handle NaN/inf inputs by returning 1.0 (no change)
        try:
            vals = [f2, f3, x2, x3, lam2, lam3, r2, r3]
            if any(not np.isfinite(v) for v in vals):
                return 1.0
        except (TypeError, ValueError):
            return 1.0
        indicator = 1.0 if (abs(r2) > 0 or abs(r3) > 0) else 0.0
        top = (f2 * (1.0 + x2) + f3 * (1.0 + x3)) * (lam2 + lam3)
        bot = lam2 * (1.0 + x2) + lam3 * (1.0 + x3)
        result = 1.0 + indicator * (top / max(bot, 1e-12) - 1.0)
        return result if np.isfinite(result) else 1.0

    @staticmethod
    def quantize_step(r, quantum):
        return round(r / quantum) * quantum

    @staticmethod
    def apply_reciprocal(prev_policy, r_step):
        # Use reciprocal fractions so increases/decreases are symmetric
        # e.g. r_step = +0.10 -> prev_policy * 1.10 (10% increase)
        #      r_step = -0.10 -> prev_policy / 1.10 (10% decrease)
        # This ensures a +10% followed by -10% returns to the original value
        try:
            if r_step >= 0:
                return prev_policy * (1.0 + r_step)
            else:
                return prev_policy / (1.0 - r_step)
        except Exception:
            return prev_policy

    # -------------------
    # Simulation & plotting
    # -------------------
    def simulate_policy(self, df: pd.DataFrame, start_policy: Optional[float] = None, sim_start_index: Optional[int] = None, simulate_first_as_rebalance: bool = True, override_first_rebalance_epoch: Optional[int] = None) -> Tuple[pd.DataFrame, Tuple[list, list], pd.DataFrame]:
        """Simulate policy on provided dataframe.

        If start_policy is provided, the simulation will begin from that policy
        level instead of self.START_POLICY. If sim_start_index is provided it
        designates the integer index in `df` where simulated (future) rows begin
        — rows with index < sim_start_index are treated as historical and will
    not be considered for policy changes. By default the first simulated row
    (index == sim_start_index) is treated as a rebalance, but callers can
    disable that by passing simulate_first_as_rebalance=False or specify an
    explicit override_first_rebalance_epoch to align the cadence.
        """
        df = df.copy()
        rp = self.REBALANCE_PERIOD

        # Determine the effective first rebalance epoch for this simulation run. Callers can
        # override it explicitly; otherwise fall back to the instance configuration (defaulting
        # to the rebalance period if unset).
        effective_first_rebalance = None
        if override_first_rebalance_epoch is not None:
            try:
                effective_first_rebalance = int(override_first_rebalance_epoch)
            except Exception:
                effective_first_rebalance = None
        if effective_first_rebalance is None:
            base_first = getattr(self, "FIRST_REBALANCE_EPOCH", None)
            if base_first is None:
                base_first = rp
            try:
                effective_first_rebalance = int(base_first)
            except Exception:
                effective_first_rebalance = None

        def _fmt_metric(val):
            if val is None:
                return "nan"
            try:
                if pd.isna(val):
                    return "nan"
            except Exception:
                pass
            try:
                return f"{float(val):.4f}"
            except Exception:
                return str(val)

        # rolling averages and derived percent changes
        df["avg_nodes_T2"] = df["nodes_T2"].rolling(rp, min_periods=rp).mean()
        df["avg_nodes_T3"] = df["nodes_T3"].rolling(rp, min_periods=rp).mean()

        df["avg_hours_per_wallet_T2"] = np.where(df["nodes_T2"] > 0, df["hours_T2"] / df["nodes_T2"], np.nan)
        df["avg_hours_per_wallet_T3"] = np.where(df["nodes_T3"] > 0, df["hours_T3"] / df["nodes_T3"], np.nan)

        df["avg_hpw_T2"] = df["avg_hours_per_wallet_T2"].rolling(rp, min_periods=rp).mean()
        df["avg_hpw_T3"] = df["avg_hours_per_wallet_T3"].rolling(rp, min_periods=rp).mean()

        df["r2"] = (df["avg_nodes_T2"] / df["avg_nodes_T2"].shift(rp)) - 1.0
        df["r3"] = (df["avg_nodes_T3"] / df["avg_nodes_T3"].shift(rp)) - 1.0
        df["x2"] = (df["avg_hpw_T2"] / df["avg_hpw_T2"].shift(rp)) - 1.0
        df["x3"] = (df["avg_hpw_T3"] / df["avg_hpw_T3"].shift(rp)) - 1.0

        # Before reliable OBhrs data (epoch 76) we don't trust work-hours percent changes.
        if "epoch" in df.columns:
            try:
                df.loc[df["epoch"] < 76, ["x2", "x3"]] = 0.0
            except Exception:
                pass

        # compute anchors using available SMA/growth and node/hour adjustments
        anchor = []
        node_multiplier_vals = []
        phi_mm_vals = []
        f2_effective_vals = []
        f3_effective_vals = []
        for _, row in df.iterrows():
            b = row.get("sma_burns")
            g = row.get("g_growth")
            if pd.isna(b):
                anchor.append(np.nan)
                node_multiplier_vals.append(np.nan)
                phi_mm_vals.append(np.nan)
                f2_effective_vals.append(np.nan)
                f3_effective_vals.append(np.nan)
                continue
            if pd.isna(g):
                g = 0.0

            current_epoch = int(row.get("epoch", 0))
            if effective_first_rebalance is not None and current_epoch == effective_first_rebalance:
                m_node = 1.0
                phi_mm = 1.0
                f2_used = self.F2
                f3_used = self.F3
                I = b * (1.0 + g) * m_node * phi_mm
            else:
                r2 = 0.0 if pd.isna(row.get("r2")) else row.get("r2")
                r3 = 0.0 if pd.isna(row.get("r3")) else row.get("r3")
                x2 = 0.0 if pd.isna(row.get("x2")) else row.get("x2")
                x3 = 0.0 if pd.isna(row.get("x3")) else row.get("x3")

                f2 = self.F2
                f3 = self.F3
                if "rolling_pct_T2" in df.columns and "rolling_pct_T3" in df.columns:
                    try:
                        pct2 = float(row.get("rolling_pct_T2") or 0.0)
                        pct3 = float(row.get("rolling_pct_T3") or 0.0)
                        denom = max(pct2 + pct3, 1e-12)
                        f2 = (pct2 / denom)
                        f3 = (pct3 / denom)
                    except Exception:
                        f2, f3 = self.F2, self.F3

                f2_used = f2
                f3_used = f3
                m_node = self.node_change_multiplier(f2_used, f3_used, r2, r3, x2, x3)
                phi_mm = self.min_max_equal_pain(f2_used, f3_used, x2, x3, self.LAMBDA2, self.LAMBDA3, r2, r3)
                I = b * (1.0 + g) * m_node * phi_mm
            anchor.append(I)
            node_multiplier_vals.append(m_node)
            phi_mm_vals.append(phi_mm)
            f2_effective_vals.append(f2_used)
            f3_effective_vals.append(f3_used)
        df["anchor_prelogic"] = anchor
        df["node_multiplier_component"] = pd.Series(node_multiplier_vals, index=df.index)
        df["phi_mm_component"] = pd.Series(phi_mm_vals, index=df.index)
        df["f2_effective_component"] = pd.Series(f2_effective_vals, index=df.index)
        df["f3_effective_component"] = pd.Series(f3_effective_vals, index=df.index)

        # simulation state
        policy = np.full(len(df), np.nan, dtype=float)
        decisions = []
        prev_policy = start_policy if start_policy is not None else self.START_POLICY
        policy[0] = prev_policy

        last_change_dir = 0
        last_cap_abs = self.NOCHANGE_BUFFER

        # default sim_start_index if not provided: simulate from the beginning (first row)
        if sim_start_index is None:
            sim_start_index = 0

        # determine first future epoch (if sim_start_index points inside df)
        first_future_epoch = None
        if 0 <= sim_start_index < len(df):
            try:
                first_future_epoch = int(df.loc[sim_start_index, "epoch"])
            except Exception:
                first_future_epoch = None

        future_anchor_epoch = None
        if first_future_epoch is not None:
            if effective_first_rebalance is not None:
                future_anchor_epoch = max(effective_first_rebalance, first_future_epoch)
            else:
                future_anchor_epoch = first_future_epoch

        # track which rows were evaluated as rebalance epochs (for plotting)
        is_rebalance_flags = [False] * len(df)

        for i, row in df.iterrows():
            epoch = int(row.get("epoch", 0))
            # Skip only if we're before the simulation start index
            if i < sim_start_index:
                policy[i] = prev_policy
                # Still need to record decisions for historical rows that are after index 0
                # (will be handled by the historical check below)
            else:
                policy[i] = prev_policy

            # compute per-row dynamic F2/F3 for reporting
            f2_row = self.F2
            f3_row = self.F3
            if "rolling_pct_T2" in df.columns and "rolling_pct_T3" in df.columns:
                try:
                    pct2 = float(row.get("rolling_pct_T2") or 0.0)
                    pct3 = float(row.get("rolling_pct_T3") or 0.0)
                    denom = max(pct2 + pct3, 1e-12)
                    f2_row = (pct2 / denom)
                    f3_row = (pct3 / denom)
                except Exception:
                    f2_row, f3_row = self.F2, self.F3

            row_sma = row.get("sma_burns")
            row_g = row.get("g_growth")
            row_r2 = row.get("r2")
            row_r3 = row.get("r3")
            row_x2 = row.get("x2")
            row_x3 = row.get("x3")
            row_anchor = row.get("anchor_prelogic")
            row_node_mult = row.get("node_multiplier_component")
            row_phi_mm = row.get("phi_mm_component")
            row_f2_eff = row.get("f2_effective_component")
            row_f3_eff = row.get("f3_effective_component")

            metrics_summary = (
                f"sma={_fmt_metric(row_sma)}, g={_fmt_metric(row_g)}, "
                f"r2={_fmt_metric(row_r2)}, r3={_fmt_metric(row_r3)}, "
                f"x2={_fmt_metric(row_x2)}, x3={_fmt_metric(row_x3)}, "
                f"node_mult={_fmt_metric(row_node_mult)}, phi_mm={_fmt_metric(row_phi_mm)}, "
                f"f2={_fmt_metric(row_f2_eff)}, f3={_fmt_metric(row_f3_eff)}"
            )

            def _decision_record(decision_type: str, reason_code: str, new_policy_val: float, cap_used_val: float, r_step_val: float) -> dict:
                return {
                    "epoch": epoch,
                    "decision": decision_type,
                    "prev_policy": prev_policy,
                    "new_policy": new_policy_val,
                    "cap_used": cap_used_val,
                    "r_step": r_step_val,
                    "reason_code": reason_code,
                    "reason": f"{reason_code} | metrics: {metrics_summary}",
                    "anchor_prelogic": row_anchor,
                    "sma_burns": row_sma,
                    "g_growth": row_g,
                    "r2": row_r2,
                    "r3": row_r3,
                    "x2": row_x2,
                    "x3": row_x3,
                    "node_multiplier": row_node_mult,
                    "phi_mm": row_phi_mm,
                    "f2_effective": row_f2_eff,
                    "f3_effective": row_f3_eff,
                    "metrics_summary": metrics_summary,
                }

            # historical rows: carry forward without attempting to rebalance
            if i < sim_start_index:
                decisions.append(_decision_record("historical", "historical_row_not_simulated", prev_policy, 0.0, 0.0))
                continue

            # Determine rebalance epochs:
            # - The first simulated row (i == sim_start_index) is always a rebalance.
            # - For historical rows (i < sim_start_index) keep original epoch%rp behavior.
            # - For future rows (i >= sim_start_index) anchor the schedule to the
            #   first future epoch so rebalances occur every REBALANCE_PERIOD from there.
            if i < sim_start_index:
                # historical rows keep original behavior
                is_rebalance_epoch = (epoch % self.REBALANCE_PERIOD == 0)
            else:
                # future rows (including the first simulated row)
                if future_anchor_epoch is not None:
                    if epoch < future_anchor_epoch:
                        is_rebalance_epoch = False
                    else:
                        if epoch == future_anchor_epoch and not simulate_first_as_rebalance:
                            is_rebalance_epoch = False
                        else:
                            is_rebalance_epoch = ((epoch - future_anchor_epoch) % self.REBALANCE_PERIOD == 0)
                elif first_future_epoch is not None:
                    diff = epoch - first_future_epoch
                    if diff == 0 and not simulate_first_as_rebalance:
                        is_rebalance_epoch = False
                    else:
                        is_rebalance_epoch = (diff % self.REBALANCE_PERIOD == 0)
                else:
                    is_rebalance_epoch = (epoch % self.REBALANCE_PERIOD == 0)
            # record flag
            if is_rebalance_epoch:
                is_rebalance_flags[i] = True
            if not is_rebalance_epoch:
                decisions.append(_decision_record("hold", "not_rebalance_epoch", prev_policy, 0.0, 0.0))
                continue

            B = row.get("sma_burns")
            I = row.get("anchor_prelogic")
            if pd.isna(B) or pd.isna(I):
                decisions.append(_decision_record("insufficient_data", "insufficient_data", prev_policy, 0.0, 0.0))
                continue

            band_base = I if not pd.isna(I) else B
            lower_band = self.BAND_MULTIPLIER_LOWER * band_base
            upper_band = self.BAND_MULTIPLIER_UPPER * band_base

            within_buffer = (abs(I - prev_policy) <= self.NOCHANGE_BUFFER * prev_policy)
            prev_within_band = (lower_band <= prev_policy <= upper_band)

            if within_buffer or prev_within_band:
                policy[i] = prev_policy
                last_change_dir = 0
                last_cap_abs = self.NOCHANGE_BUFFER
                reason = "within_buffer" if within_buffer else "prev_within_band"
                decisions.append(_decision_record("no_change", reason, prev_policy, 0.0, 0.0))
            else:
                r_raw = I / prev_policy - 1.0
                current_dir = 1 if r_raw > 0 else -1
                if last_change_dir == 0:
                    cap_abs = self.NOCHANGE_BUFFER
                else:
                    cap_abs = min(last_cap_abs + 0.10, 1.00) if current_dir == last_change_dir else self.NOCHANGE_BUFFER

                r_capped = np.clip(r_raw, -cap_abs, cap_abs)
                r_step = self.quantize_step(r_capped, self.STEP_QUANTUM)
                new_policy = self.apply_reciprocal(prev_policy, r_step)
                new_policy = int(round(new_policy / self.ROUND_TO) * self.ROUND_TO)

                policy[i] = new_policy
                last_change_dir = 0 if r_step == 0 else (1 if r_step > 0 else -1)
                last_cap_abs = cap_abs

                is_capped = not np.isclose(r_capped, r_raw)
                reason = "capped_to_limit" if is_capped else "change_to_anchor"
                decision_record = _decision_record("change", reason, new_policy, cap_abs, r_step)
                decision_record.update({"f2_dynamic": f2_row, "f3_dynamic": f3_row})
                decisions.append(decision_record)
            prev_policy = policy[i]

        df["policy"] = pd.Series(policy, index=df.index)
        # expose rebalance flags for plotting
        df["is_rebalance_epoch"] = pd.Series(is_rebalance_flags, index=df.index)

        x_step, y_step = [], []
        for i in range(len(df)):
            e = df.loc[i, "epoch"]
            p = df.loc[i, "policy"]
            if i == 0:
                x_step.extend([e, e])
                y_step.extend([p, p])
            else:
                x_step.extend([e, e])
                y_step.extend([df.loc[i - 1, "policy"], p])

        meta = pd.DataFrame(decisions)
        return df, (x_step, y_step), meta

    def make_figure(self, df: pd.DataFrame, x_step: list, y_step: list, title: str | None = None, include_trigger: bool = False, x_start_epoch: int | None = None, use_dates: bool = False) -> go.Figure:
        # Determine x-axis data: use dates if available and use_dates=True, otherwise use epoch
        if use_dates and "epoch_date" in df.columns:
            x_axis = df["epoch_date"]
            x_title = "Date"
            x_step_axis = x_step  # Note: x_step would need to be converted to dates by caller if using dates
        else:
            x_axis = df["epoch"]
            x_title = "Epoch"
            x_step_axis = x_step
        
        fig = go.Figure()
        # Use the anchor (pre-logic) as the 1× baseline and 3× = BAND_MULTIPLIER_UPPER * anchor
        fig.add_trace(go.Scatter(x=x_axis, y=df.get("anchor_prelogic", pd.Series(np.nan)), name="1× Anchor (lower band)", mode="lines", line=dict(color="rgb(240,99,90)", width=2)))
        fig.add_trace(go.Scatter(x=x_axis, y=self.BAND_MULTIPLIER_UPPER * df.get("anchor_prelogic", pd.Series(np.nan)), name="3× Anchor (upper band)", mode="lines", line=dict(color="rgb(148,193,255)", width=2)))
        # Fill between 3× anchor and 1× anchor to show the pay band area
        fig.add_trace(go.Scatter(
            x=pd.concat([x_axis, x_axis[::-1]]),
            y=pd.concat([self.BAND_MULTIPLIER_UPPER * df.get("anchor_prelogic", pd.Series(np.nan)), df.get("anchor_prelogic", pd.Series(np.nan))[::-1]]),
            fill='toself', fillcolor='rgba(148,193,255,0.15)', line=dict(color='rgba(0,0,0,0)'), showlegend=False, name="Pay band area"
        ))
        # policy stair-step
        fig.add_trace(go.Scatter(x=x_step_axis, y=y_step, name="Policy Level", mode="lines", line=dict(color="rgb(165, 243, 178)", width=3)))

        # mark rebalance epochs with open-diamond markers at the policy level
        try:
            if "is_rebalance_epoch" in df.columns:
                reb = df.loc[df["is_rebalance_epoch"].fillna(False), ["epoch", "policy"]].dropna()
                if use_dates and "epoch_date" in df.columns:
                    reb_x = df.loc[df["is_rebalance_epoch"].fillna(False), "epoch_date"]
                else:
                    reb_x = reb["epoch"]
            else:
                reb = df.loc[df["epoch"] % self.REBALANCE_PERIOD == 0, ["epoch", "policy"]].dropna()
                if use_dates and "epoch_date" in df.columns:
                    reb_x = df.loc[df["epoch"] % self.REBALANCE_PERIOD == 0, "epoch_date"]
                else:
                    reb_x = reb["epoch"]
            if not reb.empty:
                fig.add_trace(go.Scatter(
                    x=reb_x,
                    y=reb["policy"],
                    mode="markers",
                    marker=dict(symbol='diamond-open', size=10, color='yellow'),
                    name="Rebalance epochs",
                ))
        except Exception:
            # non-critical: if epoch/policy missing or invalid, skip markers
            pass

        # center title and place legend in the upper-right (outside plot) so it doesn't overlap the title
        # decide whether to use logarithmic y-axis if pay bands dwarf the policy
        yaxis_type = None
        try:
            upper_band_vals = (self.BAND_MULTIPLIER_UPPER * df.get("anchor_prelogic", pd.Series(np.nan))).dropna()
            if not upper_band_vals.empty and "policy" in df.columns:
                upper_max = float(upper_band_vals.max())
                # choose a robust policy reference (median of non-null policy values)
                pol_med = float(np.nanmedian(df["policy"])) if not df["policy"].dropna().empty else 0.0
                pol_ref = max(pol_med, 1.0)
                if upper_max / max(pol_ref, 1.0) >= 10.0:
                    yaxis_type = "log"
        except Exception:
            yaxis_type = None

        layout_kwargs = dict(
            title={"text": title or "Quarterly Policy Simulation — Stair-Step with Pay Bands", "x": 0.5, "xanchor": "center", "y": 0.98, "yanchor": "top"},
            xaxis_title=x_title,
            yaxis_title="RENDER tokens",
            template="plotly_dark",
            plot_bgcolor="black",
            paper_bgcolor="black",
            legend=dict(orientation="v", x=0.99, y=0.95, xanchor="right", yanchor="top", bgcolor='rgba(0,0,0,0)'),
            hovermode="x unified",
        )
        # Configure y-axis. If log scale chosen, build human-friendly tickvals/ticktext
        if yaxis_type is not None and yaxis_type == "log":
            # collect positive y-values to determine range
            try:
                y_candidates = []
                for col in ("policy", "anchor_prelogic",):
                    if col in df.columns:
                        vals = pd.to_numeric(df[col], errors="coerce").dropna()
                        y_candidates.extend(vals[vals > 0].tolist())
                # include upper anchor
                if "anchor_prelogic" in df.columns:
                    up = pd.to_numeric(self.BAND_MULTIPLIER_UPPER * df.get("anchor_prelogic", pd.Series(np.nan)), errors="coerce").dropna()
                    y_candidates.extend(up[up > 0].tolist())
                if len(y_candidates) == 0:
                    raise ValueError("no positive y-values for log axis")
                ymin = float(min(y_candidates))
                ymax = float(max(y_candidates))
                import math
                log_min = int(math.floor(math.log10(max(ymin, 1e-12))))
                log_max = int(math.ceil(math.log10(max(ymax, 1e-12))))
                ticks = []
                # use multipliers 0.5, 1, 2, 5 per decade for nicer intermediate ticks
                for e in range(log_min - 1, log_max + 1):
                    base = 10 ** e
                    for mul in (0.5, 1, 2, 5):
                        v = mul * base
                        if v > 0 and v >= ymin * 0.999 and v <= ymax * 1.001:
                            ticks.append(float(v))
                # ensure extremes present
                ticks = sorted(set(ticks))
                if len(ticks) == 0:
                    ticks = [10 ** log_min, 10 ** log_max]
                # format labels compactly: use k/M suffixes for thousands/millions
                def fmt_label(v):
                    if v >= 1_000_000:
                        return f"{int(round(v/1_000_000)):d}M"
                    if v >= 1000:
                        return f"{int(round(v/1000)):d}k"
                    return f"{int(round(v)):d}"

                # drop ticks that are far below the data ymin to avoid tiny integer labels
                ymin_thresh = ymin * 0.2
                ticks = [v for v in ticks if v >= ymin_thresh]
                ticktext = [fmt_label(v) for v in ticks]
                layout_kwargs["yaxis"] = dict(type="log", tickmode="array", tickvals=ticks, ticktext=ticktext)
            except Exception:
                # fallback to simple log type
                layout_kwargs["yaxis"] = dict(type="log")
        elif yaxis_type is not None:
            layout_kwargs["yaxis_type"] = yaxis_type
        # If caller supplied an explicit starting epoch for the x-axis (e.g., the first
        # future epoch after the historical tail), prefer that. Otherwise fall back to
        # the existing heuristic based on the first rebalance flag.
        try:
            if x_start_epoch is not None:
                layout_kwargs["xaxis"] = dict(range=[int(x_start_epoch), int(df["epoch"].max())])
            else:
                if "is_rebalance_epoch" in df.columns and df["is_rebalance_epoch"].any():
                    # choose the first index where is_rebalance_epoch is True AND it's a future row (policy may be nan historically)
                    mask = df["is_rebalance_epoch"].fillna(False)
                    if mask.any():
                        first_reb_epoch = int(df.loc[mask, "epoch"].iloc[0])
                        layout_kwargs["xaxis"] = dict(range=[first_reb_epoch, df["epoch"].max()])
        except Exception:
            pass

        fig.update_layout(**layout_kwargs)
        return fig

    # -------------------
    # High-level run
    # -------------------
    def run(self) -> None:
        # Preload node summary once to avoid redundant processing
        self.preload_data()

        burns = self.load_burns(self.burns_json)
        rewards = self.load_rewards(self.jobs_csv, self.avail_csv)
        tiers = self.load_tiers(self.tiers_csv)
        tiers_all = self.classify_new_wallets(rewards, tiers)

        self.out_classified.parent.mkdir(parents=True, exist_ok=True)
        tiers_all.to_csv(self.out_classified, index=False)

        tier_stats = self.per_epoch_tier_shares(rewards, tiers_all)
        df = burns.merge(tier_stats, on="epoch", how="left").sort_values("epoch").reset_index(drop=True)

        for c in ["share_T2", "share_T3"]:
            if c not in df:
                df[c] = 0.0
        for c in ["nodes_T2", "nodes_T3"]:
            if c not in df:
                df[c] = 0

        # Prefer OBhrs epoch-tier totals (from OBhrProcessing) to infer hours per tier
        obhrs_csv = self._locate_obhrs_epoch_totals()
        if obhrs_csv is not None:
            try:
                ob = pd.read_csv(obhrs_csv)
                # expect columns like epochId, T2_scaled, T3_total
                if "epochId" in ob.columns:
                    ob = ob.rename(columns={"epochId": "epoch"})
                # align epoch numbering
                if "epoch" in ob.columns:
                    ob = ob[ob["epoch"].notna()].copy()
                    ob["epoch"] = ob["epoch"].astype(int)
                    # prefer scaled T2 if present
                    t2_col = "T2_scaled" if "T2_scaled" in ob.columns else ("T2_raw" if "T2_raw" in ob.columns else None)
                    t3_col = "T3_total" if "T3_total" in ob.columns else ("T3" if "T3" in ob.columns else None)
                    # also pull rolling percent columns if present
                    pct2_col = "rolling_pct_T2" if "rolling_pct_T2" in ob.columns else ("rolling_pct_T2" if "rolling_pct_T2" in ob.columns else None)
                    pct3_col = "rolling_pct_T3" if "rolling_pct_T3" in ob.columns else ("rolling_pct_T3" if "rolling_pct_T3" in ob.columns else None)
                    keep_cols = ["epoch"]
                    if t2_col and t3_col:
                        keep_cols += [t2_col, t3_col]
                    if pct2_col:
                        keep_cols.append(pct2_col)
                    if pct3_col:
                        keep_cols.append(pct3_col)
                    # ensure unique
                    keep_cols = [c for i, c in enumerate(keep_cols) if c not in keep_cols[:i]]
                    if len(keep_cols) > 1:
                        ob = ob[keep_cols].copy()
                        # rename T2/T3 if present
                        rename_map = {}
                        if t2_col:
                            rename_map[t2_col] = "T2"
                        if t3_col:
                            rename_map[t3_col] = "T3"
                        if pct2_col:
                            rename_map[pct2_col] = "rolling_pct_T2"
                        if pct3_col:
                            rename_map[pct3_col] = "rolling_pct_T3"
                        ob = ob.rename(columns=rename_map)
                        # merge into df
                        df = df.merge(ob, on="epoch", how="left")
                        # compute hours: assume OBh -> hours mapping (maintain previous scale: *100 for T2_scaled, *200 for T3)
                        if "T2" in df.columns:
                            df["hours_T2"] = df["T2"].fillna(0.0) * 100.0
                        if "T3" in df.columns:
                            df["hours_T3"] = df["T3"].fillna(0.0) * 200.0
                        # if hours columns missing from OBhrs, ensure they exist via fallback
                        if "hours_T2" not in df:
                            df["hours_T2"] = 0.0
                        if "hours_T3" not in df:
                            df["hours_T3"] = 0.0
                    else:
                        raise ValueError("OBhrs CSV missing expected T2/T3 columns")
                else:
                    raise ValueError("OBhrs CSV missing epochId/epoch column")
            except Exception:
                # fallback to inference
                h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
                df["hours_T2"] = h2
                df["hours_T3"] = h3
        else:
            h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
            df["hours_T2"] = h2
            df["hours_T3"] = h3

        sma = df["burn_tokens"].rolling(self.REBALANCE_PERIOD, min_periods=self.REBALANCE_PERIOD).mean()
        prev = sma.shift(self.REBALANCE_PERIOD)
        g = sma / prev - 1.0
        df["sma_burns"] = sma
        df["g_growth"] = g

        df_sim, (x_step, y_step), meta = self.simulate_policy(df)

        out = df_sim[[
            "epoch", "burn_tokens", "burn_usd", "sma_burns", "g_growth",
            "share_T2", "share_T3", "nodes_T2", "nodes_T3", "hours_T2", "hours_T3",
            "avg_nodes_T2", "avg_nodes_T3", "avg_hours_per_wallet_T2", "avg_hours_per_wallet_T3",
            "avg_hpw_T2", "avg_hpw_T3", "r2", "r3", "x2", "x3", "anchor_prelogic", "policy"
        ]]

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.out_csv, index=False)
        # write decision-level report including reasons
        decisions_path = self.out_csv.parent.joinpath("policy_decisions.csv")
        try:
            meta.to_csv(decisions_path, index=False)
            print(f"Saved decisions: {decisions_path}")
        except Exception:
            pass

        fig = self.make_figure(df_sim, x_step, y_step, title="Historical Policy Simulation", include_trigger=False, x_start_epoch=1)
        self.out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(self.out_html), include_plotlyjs="cdn")

        print(f"Saved: {self.out_html}")
        print(f"Saved: {self.out_csv}")
        print(f"Saved: {self.out_classified}")

    def explain_latest_policy(self, current_policy: float | None = None) -> dict:
        """Explain the latest policy recommendation using the newest epoch's data.

        Parameters
        ----------
        current_policy : float | None, optional
            Optionally provide the currently-active policy. When omitted, the method
            will rely on the historical simulation's previous policy value.
        """
        # Preload node summary once to avoid redundant processing
        self.preload_data()

        burns = self.load_burns(self.burns_json)
        rewards = self.load_rewards(self.jobs_csv, self.avail_csv)
        tiers = self.load_tiers(self.tiers_csv)
        tiers_all = self.classify_new_wallets(rewards, tiers)

        tier_stats = self.per_epoch_tier_shares(rewards, tiers_all)
        df = burns.merge(tier_stats, on="epoch", how="left").sort_values("epoch").reset_index(drop=True)

        if df.empty:
            raise ValueError("No historical epochs available to explain a policy decision")

        for col in ["share_T2", "share_T3"]:
            if col not in df:
                df[col] = 0.0
        for col in ["nodes_T2", "nodes_T3"]:
            if col not in df:
                df[col] = 0

        obhrs_csv = self._locate_obhrs_epoch_totals()
        if obhrs_csv is not None:
            try:
                ob = pd.read_csv(obhrs_csv)
                if "epochId" in ob.columns:
                    ob = ob.rename(columns={"epochId": "epoch"})
                if "epoch" in ob.columns:
                    ob = ob[ob["epoch"].notna()].copy()
                    ob["epoch"] = ob["epoch"].astype(int)
                    t2_col = "T2_scaled" if "T2_scaled" in ob.columns else ("T2_raw" if "T2_raw" in ob.columns else None)
                    t3_col = "T3_total" if "T3_total" in ob.columns else ("T3" if "T3" in ob.columns else None)
                    pct2_col = "rolling_pct_T2" if "rolling_pct_T2" in ob.columns else None
                    pct3_col = "rolling_pct_T3" if "rolling_pct_T3" in ob.columns else None
                    keep_cols = ["epoch"]
                    if t2_col:
                        keep_cols.append(t2_col)
                    if t3_col:
                        keep_cols.append(t3_col)
                    if pct2_col:
                        keep_cols.append(pct2_col)
                    if pct3_col:
                        keep_cols.append(pct3_col)
                    keep_cols = [c for i, c in enumerate(keep_cols) if c not in keep_cols[:i]]
                    if len(keep_cols) > 1:
                        ob = ob[keep_cols].copy()
                        rename_map = {}
                        if t2_col:
                            rename_map[t2_col] = "T2"
                        if t3_col:
                            rename_map[t3_col] = "T3"
                        if pct2_col:
                            rename_map[pct2_col] = "rolling_pct_T2"
                        if pct3_col:
                            rename_map[pct3_col] = "rolling_pct_T3"
                        ob = ob.rename(columns=rename_map)
                        df = df.merge(ob, on="epoch", how="left")
                        if "T2" in df.columns:
                            df["hours_T2"] = df["T2"].fillna(0.0) * 100.0
                        if "T3" in df.columns:
                            df["hours_T3"] = df["T3"].fillna(0.0) * 200.0
                        if "hours_T2" not in df:
                            df["hours_T2"] = 0.0
                        if "hours_T3" not in df:
                            df["hours_T3"] = 0.0
                    else:
                        raise ValueError("OBhrs CSV missing expected T2/T3 columns")
                else:
                    raise ValueError("OBhrs CSV missing epoch column")
            except Exception:
                h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
                df["hours_T2"] = h2
                df["hours_T3"] = h3
        else:
            h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
            df["hours_T2"] = h2
            df["hours_T3"] = h3

        rp = int(self.REBALANCE_PERIOD)
        sma = df["burn_tokens"].rolling(rp, min_periods=rp).mean()
        prev = sma.shift(rp)
        df["sma_burns"] = sma
        df["g_growth"] = sma / prev - 1.0

        if df.empty:
            raise ValueError("Unable to locate latest epoch for explanation")

        df_hist, _, meta_hist = self.simulate_policy(df)

        hist_index = len(df_hist) - 1
        if hist_index < 0:
            raise ValueError("Unable to locate latest epoch for explanation")

        latest_epoch = int(df_hist.loc[hist_index, "epoch"])

        hist_decision_rows = meta_hist[meta_hist["epoch"] == latest_epoch] if meta_hist is not None and not meta_hist.empty else pd.DataFrame()
        hist_meta = hist_decision_rows.iloc[-1] if not hist_decision_rows.empty else None
        hist_meta_dict = hist_meta.to_dict() if hist_meta is not None else {}

        def _safe_float(val):
            try:
                if val is None:
                    return None
                if isinstance(val, (np.floating, np.integer)):
                    val = float(val)
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    return None
                if pd.isna(val):
                    return None
                return float(val)
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return None

        def _clean(val):
            if isinstance(val, (np.floating, np.integer)):
                val = float(val)
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            return val

        def _fmt_pct(val):
            val = _safe_float(val)
            if val is None:
                return "n/a"
            return f"{val:+.2%}"

        def _fmt_ratio(val):
            val = _safe_float(val)
            if val is None:
                return "n/a"
            return f"{val:.3f}"

        current_policy_level = _safe_float(current_policy) if current_policy is not None else None
        if current_policy_level is None:
            current_policy_level = _safe_float(hist_meta_dict.get("prev_policy"))
        if current_policy_level is None and hist_index >= 0:
            current_policy_level = _safe_float(df_hist.loc[hist_index, "policy"])
        if current_policy_level is None and hist_index > 0:
            current_policy_level = _safe_float(df_hist.loc[hist_index - 1, "policy"])
        if current_policy_level is None:
            current_policy_level = _safe_float(self.START_POLICY)
        if current_policy_level is None:
            raise ValueError("Unable to determine current policy for explanation")

        df_forced, _, meta_forced = self.simulate_policy(
            df,
            start_policy=float(current_policy_level),
            sim_start_index=len(df) - 1,
            simulate_first_as_rebalance=True,
        )

        row_sim = df_forced.loc[len(df_forced) - 1]
        decision_rows = meta_forced[meta_forced["epoch"] == latest_epoch] if meta_forced is not None and not meta_forced.empty else pd.DataFrame()
        decision_meta = decision_rows.iloc[-1] if not decision_rows.empty else None
        meta_dict = decision_meta.to_dict() if decision_meta is not None else {}

        burn_avg = _safe_float(meta_dict.get("sma_burns", row_sim.get("sma_burns")))
        burn_growth = _safe_float(meta_dict.get("g_growth", row_sim.get("g_growth")))
        node_multiplier = _safe_float(meta_dict.get("node_multiplier", row_sim.get("node_multiplier_component")))
        equal_pain = _safe_float(meta_dict.get("phi_mm", row_sim.get("phi_mm_component")))
        f2_eff = _safe_float(meta_dict.get("f2_effective", row_sim.get("f2_effective_component")))
        f3_eff = _safe_float(meta_dict.get("f3_effective", row_sim.get("f3_effective_component")))
        r2 = _safe_float(meta_dict.get("r2", row_sim.get("r2")))
        r3 = _safe_float(meta_dict.get("r3", row_sim.get("r3")))
        x2 = _safe_float(meta_dict.get("x2", row_sim.get("x2")))
        x3 = _safe_float(meta_dict.get("x3", row_sim.get("x3")))
        anchor = _safe_float(meta_dict.get("anchor_prelogic", row_sim.get("anchor_prelogic")))
        cap_used = _safe_float(meta_dict.get("cap_used"))
        r_step = _safe_float(meta_dict.get("r_step"))
        new_policy = _safe_float(meta_dict.get("new_policy", row_sim.get("policy")))
        prev_policy = _safe_float(meta_dict.get("prev_policy"))
        if prev_policy is None:
            prev_policy = current_policy_level
        reason_code = meta_dict.get("reason_code")
        decision_code = meta_dict.get("decision", "unknown")
        is_rebalance_epoch = True

        if new_policy is None:
            new_policy = prev_policy

        lower_band = anchor * self.BAND_MULTIPLIER_LOWER if anchor is not None else None
        upper_band = anchor * self.BAND_MULTIPLIER_UPPER if anchor is not None else None
        within_buffer = False
        within_band = False
        if anchor is not None and prev_policy is not None:
            within_buffer = abs(anchor - prev_policy) <= self.NOCHANGE_BUFFER * max(prev_policy, 1e-12)
            if lower_band is not None and upper_band is not None:
                within_band = lower_band <= prev_policy <= upper_band

        meta_clean = {k: _clean(v) for k, v in meta_dict.items()}
        details = {
            "epoch": latest_epoch,
            "burn_avg": burn_avg,
            "burn_growth": burn_growth,
            "node_multiplier": node_multiplier,
            "equal_pain": equal_pain,
            "f2_effective": f2_eff,
            "f3_effective": f3_eff,
            "r2": r2,
            "r3": r3,
            "x2": x2,
            "x3": x3,
            "anchor": anchor,
            "prev_policy": prev_policy,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "within_buffer": within_buffer,
            "within_band": within_band,
            "cap_used": cap_used,
            "r_step": r_step,
            "reason_code": reason_code,
            "is_rebalance_epoch": is_rebalance_epoch,
            "assumed_rebalance": True,
            "historical_decision": hist_meta_dict.get("decision"),
            "historical_reason_code": hist_meta_dict.get("reason_code"),
            "historical_policy": _safe_float(hist_meta_dict.get("new_policy", df_hist.loc[hist_index, "policy"] if hist_index >= 0 else None)),
            "meta": meta_clean,
        }

        reason_parts = []
        burn_avg_str = f"{burn_avg:,.0f}" if burn_avg is not None else "n/a"
        reason_parts.append(
            f"Epoch {latest_epoch} average burns over the last {self.REBALANCE_PERIOD} epochs were {burn_avg_str} tokens ({_fmt_pct(burn_growth)} growth)."
        )
        reason_parts.append(
            (
                "Node multiplier {nm} uses Tier mix f2 {f2}, f3 {f3}, node growth T2 {r2}, T3 {r3}, "
                "hours change T2 {x2}, T3 {x3}."
            ).format(
                nm=_fmt_ratio(node_multiplier),
                f2=_fmt_pct(f2_eff),
                f3=_fmt_pct(f3_eff),
                r2=_fmt_pct(r2),
                r3=_fmt_pct(r3),
                x2=_fmt_pct(x2),
                x3=_fmt_pct(x3),
            )
        )

        if anchor is None:
            reason_parts.append("Anchor could not be computed due to insufficient history; policy remains unchanged.")
            details["is_rebalance_epoch"] = is_rebalance_epoch
            return {
                "epoch": latest_epoch,
                "is_rebalance_epoch": details["is_rebalance_epoch"],
                "new_policy": prev_policy if prev_policy is not None else float(current_policy or 0.0),
                "decision": "insufficient_data",
                "reason": " ".join(reason_parts),
                "details": details,
            }

        anchor_str = f"{anchor:,.0f}" if anchor is not None else "n/a"
        reason_parts.append(
            f"Equal-pain factor {_fmt_ratio(equal_pain)} produces an anchor of {anchor_str} tokens."
        )

        if decision_code == "hold" and reason_code == "not_rebalance_epoch":
            reason_parts.append(
                f"This epoch is off-cycle for the {self.REBALANCE_PERIOD}-epoch cadence, so policy stays at {prev_policy:,.0f}."
            )
            details["is_rebalance_epoch"] = False
            return {
                "epoch": latest_epoch,
                "is_rebalance_epoch": False,
                "new_policy": prev_policy,
                "decision": "hold_not_rebalance_epoch",
                "reason": " ".join(reason_parts),
                "details": details,
            }

        if decision_code == "no_change":
            reason_parts.append(
                f"Current policy {prev_policy:,.0f} is {'within the ±{:.0%} buffer'.format(self.NOCHANGE_BUFFER) if within_buffer else 'inside the pay band'}, so no adjustment is made."
            )
            return {
                "epoch": latest_epoch,
                "is_rebalance_epoch": details["is_rebalance_epoch"],
                "new_policy": prev_policy,
                "decision": "no_change",
                "reason": " ".join(reason_parts),
                "details": details,
            }

        if decision_code == "change":
            if cap_used is not None and not np.isclose(cap_used, self.NOCHANGE_BUFFER):
                reason_parts.append(
                    f"Anchor implies a {_fmt_pct(anchor / max(prev_policy, 1e-12) - 1.0)} move but cap {cap_used:.0%} constrains the step to {_fmt_pct(r_step)}."
                )
            else:
                reason_parts.append(
                    f"Recommend adjusting policy to {new_policy:,.0f} ({_fmt_pct(r_step)} step)."
                )
            return {
                "epoch": latest_epoch,
                "is_rebalance_epoch": details["is_rebalance_epoch"],
                "new_policy": new_policy,
                "decision": "change",
                "reason": " ".join(reason_parts),
                "details": details,
            }

        if decision_code == "insufficient_data":
            reason_parts.append("Anchor inputs were incomplete, so policy remains unchanged.")
            return {
                "epoch": latest_epoch,
                "is_rebalance_epoch": details["is_rebalance_epoch"],
                "new_policy": prev_policy,
                "decision": "insufficient_data",
                "reason": " ".join(reason_parts),
                "details": details,
            }

        reason_parts.append(
            f"Decision code '{decision_code}' ({reason_code}) leaves policy at {new_policy:,.0f}."
        )
        return {
            "epoch": latest_epoch,
            "is_rebalance_epoch": details["is_rebalance_epoch"],
            "new_policy": new_policy,
            "decision": decision_code or "unknown",
            "reason": " ".join(reason_parts),
            "details": details,
        }

    def run_hypothetical(
        self,
        start_epoch: int,
        start_policy: float,
        nodes_multiplier: float | dict | None = None,
        hours_multiplier: float | dict | None = None,
    ) -> Tuple[pd.DataFrame, Tuple[list, list], pd.DataFrame]:
        """Run a hypothetical simulation applying multipliers from start_epoch onward.

        - nodes_multiplier: float to apply to both nodes_T2/nodes_T3 or dict{"T2":x, "T3":y}
        - hours_multiplier: float or dict to apply to hours_T2/hours_T3
        Returns the simulated df, steps, and meta; also writes outputs to reports with `_hypothetical` suffix.
        """
        # Preload node summary once to avoid redundant processing
        self.preload_data()

        # prepare base data same as run()
        burns = self.load_burns(self.burns_json)
        rewards = self.load_rewards(self.jobs_csv, self.avail_csv)
        tiers = self.load_tiers(self.tiers_csv)
        tiers_all = self.classify_new_wallets(rewards, tiers)

        tier_stats = self.per_epoch_tier_shares(rewards, tiers_all)
        df = burns.merge(tier_stats, on="epoch", how="left").sort_values("epoch").reset_index(drop=True)

        for c in ["share_T2", "share_T3"]:
            if c not in df:
                df[c] = 0.0
        for c in ["nodes_T2", "nodes_T3"]:
            if c not in df:
                df[c] = 0

        # copy hours from run() logic (reuse OBhrs if present)
        obhrs_csv = self._locate_obhrs_epoch_totals()
        if obhrs_csv is not None:
            try:
                ob = pd.read_csv(obhrs_csv)
                if "epochId" in ob.columns:
                    ob = ob.rename(columns={"epochId": "epoch"})
                if "epoch" in ob.columns:
                    ob["epoch"] = ob["epoch"].astype(int)
                    t2_col = "T2_scaled" if "T2_scaled" in ob.columns else ("T2_raw" if "T2_raw" in ob.columns else None)
                    t3_col = "T3_total" if "T3_total" in ob.columns else ("T3" if "T3" in ob.columns else None)
                    keep = ["epoch"]
                    if t2_col:
                        keep.append(t2_col)
                    if t3_col:
                        keep.append(t3_col)
                    ob = ob[keep].rename(columns={t2_col: "T2", t3_col: "T3"}) if len(keep) > 1 else ob
                    df = df.merge(ob, on="epoch", how="left")
                    if "T2" in df:
                        df["hours_T2"] = df["T2"].fillna(0.0) * 100.0
                    if "T3" in df:
                        df["hours_T3"] = df["T3"].fillna(0.0) * 200.0
            except Exception:
                h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
                df["hours_T2"] = h2
                df["hours_T3"] = h3
        else:
            h2, h3 = self.infer_hours_per_tier(df.get("burn_usd", pd.Series(0.0)), df["share_T2"], df["share_T3"], eurusd=self.DEFAULT_EURUSD)
            df["hours_T2"] = h2
            df["hours_T3"] = h3

        # Apply multipliers from start_epoch onward
        mask = df["epoch"] >= int(start_epoch)
        # nodes
        if nodes_multiplier is not None:
            if isinstance(nodes_multiplier, dict):
                if "T2" in nodes_multiplier:
                    df.loc[mask, "nodes_T2"] = df.loc[mask, "nodes_T2"] * float(nodes_multiplier["T2"])
                if "T3" in nodes_multiplier:
                    df.loc[mask, "nodes_T3"] = df.loc[mask, "nodes_T3"] * float(nodes_multiplier["T3"])
            else:
                df.loc[mask, "nodes_T2"] = df.loc[mask, "nodes_T2"] * float(nodes_multiplier)
                df.loc[mask, "nodes_T3"] = df.loc[mask, "nodes_T3"] * float(nodes_multiplier)
        # hours
        if hours_multiplier is not None:
            if isinstance(hours_multiplier, dict):
                if "T2" in hours_multiplier:
                    df.loc[mask, "hours_T2"] = df.loc[mask, "hours_T2"] * float(hours_multiplier["T2"])
                if "T3" in hours_multiplier:
                    df.loc[mask, "hours_T3"] = df.loc[mask, "hours_T3"] * float(hours_multiplier["T3"])
            else:
                df.loc[mask, "hours_T2"] = df.loc[mask, "hours_T2"] * float(hours_multiplier)
                df.loc[mask, "hours_T3"] = df.loc[mask, "hours_T3"] * float(hours_multiplier)

        # recompute shares from hours (simple proxy)
        total_hours = df["hours_T2"].fillna(0.0) + df["hours_T3"].fillna(0.0)
        df["share_T2"] = np.where(total_hours > 0, df["hours_T2"] / total_hours, df.get("share_T2", 0.0))
        df["share_T3"] = np.where(total_hours > 0, df["hours_T3"] / total_hours, df.get("share_T3", 0.0))

        # recompute SMAs and growth
        sma = df["burn_tokens"].rolling(self.REBALANCE_PERIOD, min_periods=self.REBALANCE_PERIOD).mean()
        prev = sma.shift(self.REBALANCE_PERIOD)
        g = sma / prev - 1.0
        df["sma_burns"] = sma
        df["g_growth"] = g

        df_sim, (x_step, y_step), meta = self.simulate_policy(df, start_policy=start_policy)

        # write outputs
        out = df_sim[[
            "epoch", "burn_tokens", "burn_usd", "sma_burns", "g_growth",
            "share_T2", "share_T3", "nodes_T2", "nodes_T3", "hours_T2", "hours_T3",
            "avg_nodes_T2", "avg_nodes_T3", "avg_hours_per_wallet_T2", "avg_hours_per_wallet_T3",
            "avg_hpw_T2", "avg_hpw_T3", "r2", "r3", "x2", "x3", "anchor_prelogic", "policy"
        ]]

        out_path = self.out_csv.parent.joinpath(self.out_csv.stem + "_hypothetical.csv")
        decisions_path = self.out_csv.parent.joinpath(self.out_csv.stem + "_hypothetical_decisions.csv")
        out.to_csv(out_path, index=False)
        try:
            meta.to_csv(decisions_path, index=False)
        except Exception:
            pass

        return df_sim, (x_step, y_step), meta

    # -------------------
    # Scenario generation helpers
    # -------------------
    def _generate_curve(self, start_value: float, target_value: float, periods: int, curve: str = "linear") -> np.ndarray:
        """Generate a series of length `periods` starting at start_value and ending at target_value.

        curve: one of 'linear', 'exponential', 's-curve'
        """
        if periods <= 1:
            return np.array([target_value])
        start = float(start_value)
        target = float(target_value)
        t = np.linspace(0.0, 1.0, periods)
        if curve == "linear":
            vals = start + (target - start) * t
        elif curve == "exponential":
            # avoid zero or negative by working in multiplier space
            if start == 0:
                vals = np.linspace(start, target, periods)
            else:
                ratio = target / start if start != 0 else 1.0
                vals = start * np.power(ratio, t)
        elif curve == "s-curve" or curve == "sigmoid":
            # logistic between 0..1 then scale
            k = 10.0
            s = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
            s = (s - s.min()) / (s.max() - s.min())
            vals = start + (target - start) * s
        else:
            # fallback to linear
            vals = start + (target - start) * t
        return vals

    def _generate_piecewise_curve(self, start_value: float, schedule, curve: str = "bezier") -> np.ndarray:
        """Generate a smooth piecewise series passing through targets defined by `schedule`.

        schedule: iterable of (multiplier, epochs) tuples. Each multiplier is interpreted
        as a multiple of the baseline `start_value` (i.e. multiplier * start_value).
        curve: currently supports 'bezier' (Catmull-Rom style smoothing) and falls back to linear.

        Returns an ndarray of length sum(epochs) containing the future values (does not include the start_value).
        """
        # validate schedule
        if not schedule:
            return np.array([])

        # normalize schedule to list of (float mult, int epochs)
        segments = []
        for item in schedule:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                mult = float(item[0])
                dur = int(item[1])
                if dur <= 0:
                    continue
                segments.append((mult, dur))
            else:
                raise ValueError("Schedule items must be (multiplier, epochs) tuples")

        baseline = float(start_value)
        # construct knots: start baseline then each segment target = baseline * mult
        knots = [baseline]
        durations = []
        for mult, dur in segments:
            knots.append(baseline * float(mult))
            durations.append(int(dur))

        total = sum(durations)
        if total == 0:
            return np.array([])

        # if only a single segment, fallback to simple curve between baseline and target
        if len(knots) == 2:
            return self._generate_curve(baseline, knots[1], total, curve=("linear" if curve != "bezier" else "s-curve"))

        values = np.zeros(total)
        idx = 0
        # Catmull-Rom style smoothing across knots (produces smooth curve through knots)
        for i in range(len(knots) - 1):
            p1 = knots[i]
            p2 = knots[i + 1]
            p0 = knots[i - 1] if i - 1 >= 0 else p1
            p3 = knots[i + 2] if i + 2 < len(knots) else p2
            dur = durations[i]
            for j in range(1, dur + 1):
                t = j / float(dur)
                if curve == "bezier" or curve == "catmull-rom":
                    # Catmull-Rom spline (centripetal parameterization omitted for simplicity)
                    t2 = t * t
                    t3 = t2 * t
                    val = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
                else:
                    # fallback linear interpolation within this segment
                    val = p1 + (p2 - p1) * t
                if idx < total:
                    values[idx] = val
                    idx += 1

        return values

    def _build_future_dataframe(self, base_df: pd.DataFrame, periods: int, base: str = "burn", curve: str = "linear", multiplier: float = 1.0, hist_rows: int | None = None) -> pd.DataFrame:
        """Build a DataFrame that appends `periods` future epochs to the latest epoch in base_df.

        base: 'burn' | 'node' | 'hybrid'
        curve: 'linear' | 'exponential' | 's-curve'
        multiplier: overall multiplier to apply to the varied series (interpreted per-base)
        hist_rows: number of historical rows to include before futures (defaults to REBALANCE_PERIOD)
        """
        df_hist = base_df.copy().sort_values("epoch").reset_index(drop=True)
        if df_hist.empty:
            raise ValueError("Base dataframe is empty; cannot generate future scenario.")

        last = df_hist.iloc[-1]
        start_epoch = int(last.get("epoch", 0))

        # seed values: prefer rolling SMA over raw last value so future sims start from
        # the smoothed baseline. Use min_periods=1 so SMA is available with limited history.
        rp = int(self.REBALANCE_PERIOD)
        try:
            sma_burn = df_hist["burn_tokens"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_burn = None
        try:
            sma_burn_usd = df_hist["burn_usd"].rolling(rp, min_periods=1).mean().iloc[-1] if "burn_usd" in df_hist.columns else None
        except Exception:
            sma_burn_usd = None

        try:
            sma_nodes_T2 = df_hist["nodes_T2"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_nodes_T2 = None
        try:
            sma_nodes_T3 = df_hist["nodes_T3"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_nodes_T3 = None

        try:
            sma_hours_T2 = df_hist["hours_T2"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_hours_T2 = None
        try:
            sma_hours_T3 = df_hist["hours_T3"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_hours_T3 = None

        try:
            sma_share_T2 = df_hist["share_T2"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_share_T2 = None
        try:
            sma_share_T3 = df_hist["share_T3"].rolling(rp, min_periods=1).mean().iloc[-1]
        except Exception:
            sma_share_T3 = None

        # fallback to raw last values if SMA not available
        last_burn = float(sma_burn if sma_burn is not None and not pd.isna(sma_burn) else (last.get("burn_tokens") or 0.0))
        last_burn_usd = float(sma_burn_usd if sma_burn_usd is not None and not pd.isna(sma_burn_usd) else (last.get("burn_usd") or 0.0))
        last_nodes_T2 = float(sma_nodes_T2 if sma_nodes_T2 is not None and not pd.isna(sma_nodes_T2) else (last.get("nodes_T2") or 0.0))
        last_nodes_T3 = float(sma_nodes_T3 if sma_nodes_T3 is not None and not pd.isna(sma_nodes_T3) else (last.get("nodes_T3") or 0.0))
        last_hours_T2 = float(sma_hours_T2 if sma_hours_T2 is not None and not pd.isna(sma_hours_T2) else (last.get("hours_T2") or 0.0))
        last_hours_T3 = float(sma_hours_T3 if sma_hours_T3 is not None and not pd.isna(sma_hours_T3) else (last.get("hours_T3") or 0.0))
        last_share_T2 = float(sma_share_T2 if sma_share_T2 is not None and not pd.isna(sma_share_T2) else (last.get("share_T2") or 0.0))
        last_share_T3 = float(sma_share_T3 if sma_share_T3 is not None and not pd.isna(sma_share_T3) else (last.get("share_T3") or 0.0))

        # support schedule multipliers: if multiplier is an iterable of tuples, generate piecewise series
        is_schedule = False
        schedule_total_epochs = None
        try:
            is_schedule = (hasattr(multiplier, "__iter__") and not isinstance(multiplier, (str, bytes)) and not isinstance(multiplier, (int, float)))
            if is_schedule:
                # compute total duration requested by the schedule so we can respect it
                try:
                    total = 0
                    for item in multiplier:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            dur = int(item[1])
                            if dur > 0:
                                total += dur
                    schedule_total_epochs = total if total > 0 else None
                    # if the schedule requests more epochs than `periods`, prefer the schedule
                    if schedule_total_epochs is not None and schedule_total_epochs > int(periods):
                        periods = int(schedule_total_epochs)
                except Exception:
                    schedule_total_epochs = None
        except Exception:
            is_schedule = False

        # decide targets depending on base
        if base == "burn":
            if is_schedule:
                # multiplier is a schedule of (mult, epochs)
                burn_series = self._generate_piecewise_curve(last_burn, multiplier, curve=("bezier" if curve == "bezier" else "s-curve"))
                # ensure length matches periods
                if len(burn_series) > periods:
                    burn_series = burn_series[:periods]
                elif len(burn_series) < periods:
                    # pad with last value
                    burn_series = np.concatenate([burn_series, np.full(periods - len(burn_series), burn_series[-1] if len(burn_series) else last_burn)])
            else:
                target_burn = last_burn * float(multiplier)
                burn_series = self._generate_curve(last_burn, target_burn, periods, curve)
            # keep nodes/hours stable at last observed
            nodes_T2_series = np.full(periods, last_nodes_T2)
            nodes_T3_series = np.full(periods, last_nodes_T3)
            hours_T2_series = np.full(periods, last_hours_T2)
            hours_T3_series = np.full(periods, last_hours_T3)
            burn_usd_series = np.full(periods, last_burn_usd * float(multiplier) if not is_schedule else last_burn_usd)
        elif base == "node":
            # apply multiplier to nodes and hours
            if is_schedule:
                nodes_vals = self._generate_piecewise_curve(last_nodes_T2, multiplier, curve=("bezier" if curve == "bezier" else "s-curve"))
                # apply same schedule to both tiers proportionally if nodes_vals length matches
                if len(nodes_vals) >= periods:
                    nodes_T2_series = nodes_vals[:periods]
                    nodes_T3_series = nodes_vals[:periods]
                else:
                    nodes_T2_series = np.concatenate([nodes_vals, np.full(periods - len(nodes_vals), nodes_vals[-1] if len(nodes_vals) else last_nodes_T2)])
                    nodes_T3_series = nodes_T2_series.copy()
                # hours follow nodes
                hours_T2_series = (nodes_T2_series / np.where(last_nodes_T2 == 0, 1, last_nodes_T2)) * last_hours_T2
                hours_T3_series = (nodes_T3_series / np.where(last_nodes_T3 == 0, 1, last_nodes_T3)) * last_hours_T3
            else:
                target_nodes_T2 = last_nodes_T2 * float(multiplier)
                target_nodes_T3 = last_nodes_T3 * float(multiplier)
                nodes_T2_series = self._generate_curve(last_nodes_T2, target_nodes_T2, periods, curve)
                nodes_T3_series = self._generate_curve(last_nodes_T3, target_nodes_T3, periods, curve)
                # hours follow nodes by default (proportional)
                target_hours_T2 = last_hours_T2 * float(multiplier)
                target_hours_T3 = last_hours_T3 * float(multiplier)
                hours_T2_series = self._generate_curve(last_hours_T2, target_hours_T2, periods, curve)
                hours_T3_series = self._generate_curve(last_hours_T3, target_hours_T3, periods, curve)
            # keep burns stable as baseline
            burn_series = np.full(periods, last_burn)
            burn_usd_series = np.full(periods, last_burn_usd)
        elif base == "hybrid":
            # apply half the multiplier to burns and half to nodes to provide a combined effect
            if is_schedule:
                # expand schedule twice: first half influences burns, second influences nodes proportionally
                burn_vals = self._generate_piecewise_curve(last_burn, multiplier, curve=("bezier" if curve == "bezier" else "s-curve"))
                if len(burn_vals) >= periods:
                    burn_series = burn_vals[:periods]
                else:
                    burn_series = np.concatenate([burn_vals, np.full(periods - len(burn_vals), burn_vals[-1] if len(burn_vals) else last_burn)])
                # use same schedule for nodes
                node_vals = burn_series
                nodes_T2_series = (node_vals / np.where(last_nodes_T2 == 0, 1, last_nodes_T2)) * last_nodes_T2
                nodes_T3_series = (node_vals / np.where(last_nodes_T3 == 0, 1, last_nodes_T3)) * last_nodes_T3
                hours_T2_series = (nodes_T2_series / np.where(last_nodes_T2 == 0, 1, last_nodes_T2)) * last_hours_T2
                hours_T3_series = (nodes_T3_series / np.where(last_nodes_T3 == 0, 1, last_nodes_T3)) * last_hours_T3
                burn_usd_series = np.full(periods, last_burn_usd)
            else:
                target_burn = last_burn * float(1.0 + (multiplier - 1.0) * 0.5)
                burn_series = self._generate_curve(last_burn, target_burn, periods, curve)
                target_nodes_T2 = last_nodes_T2 * float(1.0 + (multiplier - 1.0) * 0.5)
                target_nodes_T3 = last_nodes_T3 * float(1.0 + (multiplier - 1.0) * 0.5)
                nodes_T2_series = self._generate_curve(last_nodes_T2, target_nodes_T2, periods, curve)
                nodes_T3_series = self._generate_curve(last_nodes_T3, target_nodes_T3, periods, curve)
                # hours scale with nodes
                hours_T2_series = (nodes_T2_series / np.where(last_nodes_T2 == 0, 1, last_nodes_T2)) * last_hours_T2
                hours_T3_series = (nodes_T3_series / np.where(last_nodes_T3 == 0, 1, last_nodes_T3)) * last_hours_T3
                burn_usd_series = np.full(periods, last_burn_usd)
        else:
            raise ValueError(f"Unknown base for scenario: {base}")

        # now that `periods` may have been adjusted (e.g., schedule total), build future epoch list
        future_epochs = [start_epoch + i for i in range(1, periods + 1)]

        rows = []
        for i, ep in enumerate(future_epochs):
            rows.append({
                "epoch": ep,
                "burn_tokens": float(burn_series[i]) if burn_series is not None else float(last_burn),
                "burn_usd": float(burn_usd_series[i]) if burn_usd_series is not None else float(last_burn_usd),
                "share_T2": np.nan,
                "share_T3": np.nan,
                "nodes_T2": float(nodes_T2_series[i]),
                "nodes_T3": float(nodes_T3_series[i]),
                "hours_T2": float(hours_T2_series[i]),
                "hours_T3": float(hours_T3_series[i]),
            })

        df_future = pd.DataFrame(rows)

        # Determine how many historical rows to include so rolling SMA/growth and
        # other rolling stats can be seeded from real data. Default to REBALANCE_PERIOD
        # (e.g., 12) but don't exceed available history.
        if hist_rows is None:
            hist_rows = min(len(df_hist), int(self.REBALANCE_PERIOD))
        else:
            hist_rows = max(0, min(int(hist_rows), len(df_hist)))

        hist_tail = df_hist.iloc[-hist_rows:][[
            "epoch", "burn_tokens", "burn_usd", "share_T2", "share_T3", "nodes_T2", "nodes_T3", "hours_T2", "hours_T3"
        ]].reset_index(drop=True)

        # Combine selected historical tail rows + future rows
        df_combined = pd.concat([hist_tail, df_future], ignore_index=True, sort=False)

        # Fill share columns by recomputing from hours when possible
        total_hours = df_combined["hours_T2"].fillna(0.0) + df_combined["hours_T3"].fillna(0.0)
        df_combined["share_T2"] = np.where(total_hours > 0, df_combined["hours_T2"] / total_hours, df_combined.get("share_T2", 0.0))
        df_combined["share_T3"] = np.where(total_hours > 0, df_combined["hours_T3"] / total_hours, df_combined.get("share_T3", 0.0))

        # Ensure numeric columns exist for downstream processing
        for c in ["avg_nodes_T2", "avg_nodes_T3", "avg_hours_per_wallet_T2", "avg_hours_per_wallet_T3", "avg_hpw_T2", "avg_hpw_T3", "r2", "r3", "x2", "x3"]:
            if c not in df_combined:
                df_combined[c] = np.nan

        return df_combined

    def run_scenario(self, name: str, base: str = "burn", curve: str = "linear", periods: int = 24, multiplier: float = 1.0, start_policy: float = 15000.0, first_rebalance_epoch: Optional[int] = None) -> Tuple[pd.DataFrame, Tuple[list, list], pd.DataFrame]:
        """Create and run a single future scenario.

        - name: label for outputs (used in filenames when writing)
        - base: 'burn'|'node'|'hybrid'
        - curve: 'linear'|'exponential'|'s-curve'
        - periods: number of future epochs to simulate
        - multiplier: final multiplier applied to the varied metric
        - start_policy: policy level to start the simulation from (defaults to 15000)
                - first_rebalance_epoch: optional absolute epoch for the first post-history rebalance;
                    defaults to the first new epoch when omitted

        Returns df_sim, steps, meta.
        """
        # Preload node summary once to avoid redundant processing
        self.preload_data()

        # build history to seed scenario
        burns = self.load_burns(self.burns_json)
        rewards = self.load_rewards(self.jobs_csv, self.avail_csv)
        tiers = self.load_tiers(self.tiers_csv)
        tiers_all = self.classify_new_wallets(rewards, tiers)

        tier_stats = self.per_epoch_tier_shares(rewards, tiers_all)
        df_hist = burns.merge(tier_stats, on="epoch", how="left").sort_values("epoch").reset_index(drop=True)

        for c in ["share_T2", "share_T3"]:
            if c not in df_hist:
                df_hist[c] = 0.0
        for c in ["nodes_T2", "nodes_T3"]:
            if c not in df_hist:
                df_hist[c] = 0

        # If OBhrs present, prefer those hours
        obhrs_csv = self._locate_obhrs_epoch_totals()
        if obhrs_csv is not None:
            try:
                ob = pd.read_csv(obhrs_csv)
                if "epochId" in ob.columns:
                    ob = ob.rename(columns={"epochId": "epoch"})
                if "epoch" in ob.columns:
                    ob["epoch"] = ob["epoch"].astype(int)
                    t2_col = "T2_scaled" if "T2_scaled" in ob.columns else ("T2_raw" if "T2_raw" in ob.columns else None)
                    t3_col = "T3_total" if "T3_total" in ob.columns else ("T3" if "T3" in ob.columns else None)
                    keep = ["epoch"]
                    if t2_col:
                        keep.append(t2_col)
                    if t3_col:
                        keep.append(t3_col)
                    if len(keep) > 1:
                        ob = ob[keep].rename(columns={t2_col: "T2", t3_col: "T3"})
                        df_hist = df_hist.merge(ob, on="epoch", how="left")
                        if "T2" in df_hist:
                            df_hist["hours_T2"] = df_hist["T2"].fillna(0.0) * 100.0
                        if "T3" in df_hist:
                            df_hist["hours_T3"] = df_hist["T3"].fillna(0.0) * 200.0
            except Exception:
                h2, h3 = self.infer_hours_per_tier(df_hist.get("burn_usd", pd.Series(0.0)), df_hist["share_T2"], df_hist["share_T3"], eurusd=self.DEFAULT_EURUSD)
                df_hist["hours_T2"] = h2
                df_hist["hours_T3"] = h3
        else:
            h2, h3 = self.infer_hours_per_tier(df_hist.get("burn_usd", pd.Series(0.0)), df_hist["share_T2"], df_hist["share_T3"], eurusd=self.DEFAULT_EURUSD)
            df_hist["hours_T2"] = h2
            df_hist["hours_T3"] = h3

        # Build future dataframe using the full historical data so all rolling stats
        # (SMA/growth/anchors) can be computed immediately from real values.
        df_combined = self._build_future_dataframe(df_hist, periods=periods, base=base, curve=curve, multiplier=multiplier, hist_rows=len(df_hist))

        # recompute SMA/growth across combined so SMA exists even at early epochs
        # use min_periods=1 to populate SMA from available history, then compute growth
        sma = df_combined["burn_tokens"].rolling(self.REBALANCE_PERIOD, min_periods=1).mean()
        prev_rp = sma.shift(self.REBALANCE_PERIOD)
        prev_1 = sma.shift(1)
        prev = prev_rp.fillna(prev_1)
        g = np.where(prev > 0, sma / prev - 1.0, 0.0)
        df_combined["sma_burns"] = sma
        df_combined["g_growth"] = g

        # run the simulation starting from the provided start policy (default 15000)
        # futures start at index = number of historical rows included
        sim_start_index = len(df_hist)

        # Determine the override for the first rebalance epoch used by simulate_policy.
        # - If caller provided first_rebalance_epoch explicitly, use that absolute epoch.
        # - If caller did not provide it (None), default to the first future epoch so
        #   the first rebalance will occur on the first new epoch (subject to cadence).
        override_first = None
        try:
            if first_rebalance_epoch is not None:
                override_first = int(first_rebalance_epoch)
            else:
                # first future epoch is at sim_start_index in the combined frame
                if 0 <= sim_start_index < len(df_combined):
                    override_first = int(df_combined.loc[sim_start_index, "epoch"])
        except Exception:
            override_first = None

        df_sim, steps, meta = self.simulate_policy(
            df_combined,
            start_policy=float(start_policy),
            sim_start_index=sim_start_index,
            simulate_first_as_rebalance=True,
            override_first_rebalance_epoch=override_first,
        )

        # write outputs labelled by name
        out = df_sim[[
            "epoch", "burn_tokens", "burn_usd", "sma_burns", "g_growth",
            "share_T2", "share_T3", "nodes_T2", "nodes_T3", "hours_T2", "hours_T3",
            "avg_nodes_T2", "avg_nodes_T3", "avg_hours_per_wallet_T2", "avg_hours_per_wallet_T3",
            "avg_hpw_T2", "avg_hpw_T3", "r2", "r3", "x2", "x3", "anchor_prelogic", "policy"
        ]]

        out_dir = self.out_csv.parent.joinpath("future_sims")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir.joinpath(self.out_csv.stem + f"_{name}.csv")
        decisions_path = out_dir.joinpath(self.out_csv.stem + f"_{name}_decisions.csv")
        out.to_csv(out_path, index=False)
        try:
            meta.to_csv(decisions_path, index=False)
        except Exception:
            pass

        # Also write an HTML plot per scenario into the future_sims folder
        try:
            # determine first plotted epoch for the scenario (first future epoch after historical tail)
            x_start = None
            try:
                if 0 <= sim_start_index < len(df_sim):
                    x_start = int(df_sim.loc[sim_start_index, "epoch"])
            except Exception:
                x_start = None
            fig = self.make_figure(df_sim, steps[0], steps[1], title=f"Future Policy Simulation {name}", include_trigger=False, x_start_epoch=x_start)
            html_path = out_dir.joinpath(self.out_html.stem + f"_{name}.html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        except Exception:
            html_path = None

        return df_sim, steps, meta

    def run_scenarios(self, periods: int = 24, multiplier: float = 1.2, start_policy: float = 15000.0) -> dict:
        """Run the three canonical scenarios (burn, node, hybrid) and return a dict of results.

        Returns: { 'burn': (df_sim, steps, meta), 'node': (...), 'hybrid': (...) }
        """
        results = {}
        for base in ["burn", "node", "hybrid"]:
            name = f"scenario_{base}"
            df_sim, steps, meta = self.run_scenario(name=name, base=base, curve="linear", periods=periods, multiplier=multiplier, start_policy=start_policy)
            results[base] = (df_sim, steps, meta)
        return results


if __name__ == "__main__":
    # Set skip_node_summary_regen=True to use existing node_summary.csv without regenerating it
    # This is much faster for repeat runs - only regenerate when source data (work.csv/avail.csv) changes
    sim = PolicySimulation(skip_node_summary_regen=True)
    # run the historical/default output
    sim.run()
    # give the latest policy explanation
    print(sim.explain_latest_policy(15000))

    # ============================================================================
    # EXAMPLE FUTURE SCENARIOS
    # ============================================================================
    # Uncomment any of these to run future projections and see how the policy
    # responds to different market conditions. Results are saved to reports/future_sims/
    # All scenarios use the current policy level as the starting point.
    
    # ----------------------------------------------------------------------------
    # BURN-BASED SCENARIOS (Demand-Driven) SCENARIOS 2 and 3 are featured the in the RNP
    # ----------------------------------------------------------------------------
    # These scenarios focus on how the policy responds to changes in network usage/burns
    
    # Scenario 1: "Steady Growth" - Healthy, sustainable network growth
    # - Burns increase 50% over 1 year (52 epochs)
    # - Shows how policy responds to gradual, predictable growth
    # - Demonstrates the 10% cap preventing rapid jumps
    # - Expected: Policy increases gradually but steadily
    # sim.run_scenario(name="steady_growth", base="burn", curve="linear", periods=52, multiplier=1.5, start_policy=15000.0)
    
    # Scenario 2: "Market Downturn" - Bear market/recession scenario
    # - Burns decline 60% over 1 year (52 epochs)
    # - Shows how policy protects nodes during downturns
    # - Demonstrates controlled decreases and no-change buffer
    # - Expected: Policy decreases in controlled steps, may hit buffer frequently
    # sim.run_scenario(name="market_downturn", base="burn", curve="linear", periods=52, multiplier=0.4, start_policy=15000.0)
    
    # Scenario 3: "Explosive Growth" - Viral adoption/major bull run
    # - Burns 3x (200% increase) over 9 months (36 epochs)
    # - Shows how policy handles rapid, accelerating growth
    # - Demonstrates importance of 10% cap and cap escalation
    # - Expected: Policy increases aggressively but in controlled steps insuring burns exceed issuance while paying ops fairly
    # sim.run_scenario(name="explosive_growth", base="burn", curve="exponential", periods=36, multiplier=3.0, start_policy=15000.0)
    
    # ----------------------------------------------------------------------------
    # NODE-BASED SCENARIOS (Supply-Driven)
    # ----------------------------------------------------------------------------
    # These scenarios focus on how the policy responds to changes in node participation
    
    # Scenario 4: "Node Expansion" - Organic network growth
    # - Node count/activity increases 50% over 1 year (52 epochs)
    # - Shows how policy responds to growing supply capacity
    # - Demonstrates the node multiplier component in action
    # - Expected: Policy increases as node capacity grows
    # sim.run_scenario(name="node_expansion", base="node", curve="linear", periods=52, multiplier=1.5, start_policy=1500.0)
    
    # Scenario 5: "Node Attrition" - Operator exodus/difficult period
    # - Node count/activity declines 35% over 1 year (52 epochs)
    # - Shows how policy protects against capacity loss
    # - Demonstrates adaptive response to supply-side constraints
    # - Expected: Policy decreases to match reduced capacity
    # sim.run_scenario(name="node_attrition", base="node", curve="linear", periods=52, multiplier=0.65, start_policy=1500.0)
    
    # Scenario 6: "Tier Migration" - Quality upgrade (T2→T3 shift)
    # - Nodes migrate from T2 to T3 tier over 10 months (40 epochs)
    # - Shows how policy handles compositional changes
    # - Demonstrates equal-pain factor and dynamic tier weights
    # - Expected: Policy adjusts based on changing tier mix
    # - Note: Use run_hypothetical() for tier-specific multipliers
    # sim_tier = PolicySimulation(skip_node_summary_regen=True)
    # df, steps, meta = sim_tier.run_hypothetical(
    #     start_epoch=97, 
    #     start_policy=1500.0,
    #     nodes_multiplier={"T2": 0.7, "T3": 1.4}  # T2 -30%, T3 +40%
    # )
    
    # Scenario 7: "Activity Surge" - Existing nodes work harder
    # - Node activity (hours worked) doubles over 9 months (36 epochs)
    # - Shows how increased utilization affects the anchor
    # - Demonstrates hours-change multiplier component
    # - Expected: Policy increases as nodes work harder
    # sim.run_scenario(name="activity_surge", base="node", curve="exponential", periods=36, multiplier=2.0, start_policy=1500.0)
    
    # ----------------------------------------------------------------------------
    # CUSTOM CADENCE EXAMPLE
    # ----------------------------------------------------------------------------
    # You can also change the rebalancing cadence (epochs per rebalance)
    # Example: 8-epoch cadence instead of default 12
    # sim_custom = PolicySimulation(rebalance_period=8, first_rebalance_epoch=99, skip_node_summary_regen=True)
    # sim_custom.run_scenario(name="steady_growth_8ep", base="burn", curve="linear", periods=52, multiplier=1.5, start_policy=1500.0)

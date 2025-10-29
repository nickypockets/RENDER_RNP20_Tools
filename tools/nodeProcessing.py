"""Node processing utilities.

Produce a single-row-per-wallet dataset that records the first epoch a wallet
appears in either the availability or work datasets, and stores each epoch's
combined rewards in its own column.

Functions:
- aggregate_node_rewards(avail_csv, work_csv, output_csv) -> pd.DataFrame
  Reads the two CSVs, sums rewards per wallet+epoch, pivots epochs into
  columns named "epoch_{epoch}" and writes the result to `output_csv`.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# ---- Epoch mapping boundary handling (fix) ----
# Some payouts arrive right after the epoch rolls. Shift timestamps back a bit
# before bucketing so they land in the intended epoch.
EPOCH_GRACE_HOURS = 3  # adjust if needed

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name in df that matches any candidate (case-insensitive).

    candidates are substrings to search for in lower-cased column names.
    """
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for low, orig in cols.items():
            if cand in low:
                return orig
    return None


def aggregate_node_rewards(
    avail_csv: str | Path = None,
    work_csv: str | Path = None,
    output_csv: str | Path = None,
) -> pd.DataFrame:
    """Aggregate avail+work rewards per Recipient Wallet and epoch.

    Output DataFrame has one row per wallet, a `first_epoch` column and one
    column per epoch named `epoch_{epoch}` containing the summed rewards for
    that wallet+epoch (float). Missing epoch values are filled with 0.

    Parameters
    - avail_csv, work_csv: paths to the two input CSVs.
    - output_csv: path to write the resulting CSV.

    Returns the resulting pandas DataFrame.
    """
    # Get project root (go up one level from tools/)
    project_root = Path(__file__).resolve().parent.parent
    
    # Use defaults if not provided
    if avail_csv is None:
        avail_csv = project_root / "data" / "avail_data" / "avail.csv"
    else:
        avail_csv = Path(avail_csv)
    
    if work_csv is None:
        work_csv = project_root / "data" / "work_data" / "work.csv"
    else:
        work_csv = Path(work_csv)
    
    if output_csv is None:
        output_csv = project_root / "data" / "node_summary.csv"
    else:
        output_csv = Path(output_csv)

    # read files if they exist, otherwise create empty DataFrames
    if avail_csv.exists():
        df_av = pd.read_csv(avail_csv)
    else:
        df_av = pd.DataFrame()

    if work_csv.exists():
        df_work = pd.read_csv(work_csv)
    else:
        df_work = pd.DataFrame()

    # try to import _ensure_epoch_end from tools.downloadData if available
    try:
        from tools.downloadData import _ensure_epoch_end

        if not df_av.empty:
            df_av = _ensure_epoch_end(df_av.copy())
        if not df_work.empty:
            df_work = _ensure_epoch_end(df_work.copy())
    except Exception:
        # fallback: assume epoch_end column already present or rows include 'Date' that pd.to_datetime can parse
        pass

    # helper to normalize DataFrame to columns: recipient_wallet, epoch_end, rewards
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["recipient_wallet", "epoch_date", "rewards"])

        # --- recipient wallet ---
        recip_col = _find_col(df, ["recipient wallet", "recipient_wallet", "recipientwallet"])
        if recip_col is None:
            raise ValueError("Could not find recipient wallet column in input dataframe")

        # --- epoch column: prefer explicit date columns over epoch_end ---
        # Many data sources have unreliable epoch_end values (like 1.0 or unix timestamps),
        # so we prefer to use actual date columns when available.
        epoch_col = _find_col(df, ["date", "blocktime"])  # Try date columns first
        if epoch_col is None:
            # Fall back to epoch columns only if no date column exists
            epoch_col = _find_col(df, ["epoch", "epoch_end", "epochend"])
            if epoch_col is None:
                raise ValueError("Could not find epoch column (epoch/epoch_end/date/blocktime) in input dataframe")

        # --- rewards column ---
        rewards_col = _find_col(df, ["$render", "render rewards", "render"])
        if rewards_col is None:
            candidates = [c for c in df.columns if c not in (recip_col, epoch_col)]
            numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
            if numeric:
                rewards_col = numeric[0]
            else:
                raise ValueError("Could not find rewards column in input dataframe")

        out = df[[recip_col, epoch_col, rewards_col]].copy()
        out.columns = ["recipient_wallet", "epoch_raw", "rewards"]
        out["recipient_wallet"] = out["recipient_wallet"].astype(str).str.lower().str.strip()

        # parse a datetime we can use later (keep as full timestamp if available)
        # we DO NOT drop to midnight here; we keep as precise as possible
        # If epoch_raw is numeric unix seconds, parse with unit='s'; otherwise generic parse.
        parsed_unit = pd.to_datetime(out["epoch_raw"], unit="s", errors="coerce", utc=True)
        parsed_generic = pd.to_datetime(out["epoch_raw"], errors="coerce", utc=True)
        parsed = parsed_unit.where(parsed_unit.notna(), parsed_generic)

        out["epoch_dt"] = parsed
        out["epoch_date"] = parsed.dt.normalize().dt.date.astype(str)  # convenience label
        out = out.dropna(subset=["recipient_wallet", "epoch_date"])

        # If the source actually contains small integer epoch IDs, preserve them.
        try:
            numeric_epoch = pd.to_numeric(df[epoch_col], errors="coerce")
            if numeric_epoch.notna().sum() > 0 and (numeric_epoch.max() < 10000):
                out["epoch"] = numeric_epoch.astype(pd.Int64Dtype())
        except Exception:
            pass

        out["rewards"] = pd.to_numeric(out["rewards"], errors="coerce").fillna(0.0)
        return out

    n_av = _normalize(df_av)
    n_work = _normalize(df_work)

    # Deduplicate records: work.csv/avail.csv may contain duplicate transactions
    # Group by (recipient_wallet, epoch_date, rewards) and keep only one copy
    if not n_av.empty:
        n_av = n_av.drop_duplicates(subset=['recipient_wallet', 'epoch_date', 'rewards'], keep='first')
    if not n_work.empty:
        n_work = n_work.drop_duplicates(subset=['recipient_wallet', 'epoch_date', 'rewards'], keep='first')

    # build node set from both avail and work recipients (lowercased).
    # Previously we only used avail recipients which dropped wallets that
    # only appear in the work dataset and produced impossible zero-work
    # epochs. Use the union so any wallet that received work is included.
    node_set = set()
    if not n_av.empty:
        node_set |= set(n_av['recipient_wallet'].dropna().unique())
    if not n_work.empty:
        node_set |= set(n_work['recipient_wallet'].dropna().unique())

    if n_av.empty and n_work.empty:
        # produce empty DataFrame with basic columns
        res = pd.DataFrame(columns=["recipient_wallet", "first_epoch"]) 
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(output_csv, index=False)
        return res


    # Build a chronological list of unique dates. Prefer the canonical
    # epoch dates found in data/burns_data/burns.json (if present). This
    # produces a universal mapping we can bucket node_summary into. If the
    # burns file is missing or doesn't contain parseable dates, fall back
    # to using the union of dates from the avail/work inputs.
    # Build canonical epoch ends from burns.json (keep full UTC timestamps)
    burn_dates_are_ends = False
    epoch_end_ts_list = []  # list[pd.Timestamp] in UTC
    project_root = Path(__file__).resolve().parent.parent
    burns_path = project_root / "data" / "burns_data" / "burns.json"
    if burns_path.exists():
        try:
            with open(burns_path, "r", encoding="utf-8") as fh:
                burns_obj = json.load(fh)

            if isinstance(burns_obj, dict) and isinstance(burns_obj.get("data"), list):
                entries = burns_obj["data"]
                # sort deterministically
                def _key(e):
                    if isinstance(e, dict) and "id" in e:
                        return e["id"]
                    if isinstance(e, dict) and "startDate" in e:
                        return pd.to_datetime(e["startDate"], utc=True, errors="coerce")
                    return 0
                entries_sorted = sorted(entries, key=_key)

                ends_ts = []
                for ent in entries_sorted:
                    if not isinstance(ent, dict):
                        continue
                    ed = ent.get("endDate") or ent.get("end_date") or ent.get("end")
                    if not ed:
                        continue
                    dt = pd.to_datetime(ed, utc=True, errors="coerce")
                    if pd.notna(dt):
                        ends_ts.append(dt)

                # If inputs contain dates later than the last canonical end, append
                try:
                    latest_input = None
                    if 'n_av' in locals() and not n_av.empty and 'epoch_dt' in n_av.columns:
                        latest_input = n_av['epoch_dt'].max()
                    if 'n_work' in locals() and not n_work.empty and 'epoch_dt' in n_work.columns:
                        wmax = n_work['epoch_dt'].max()
                        if latest_input is None or (pd.notna(wmax) and wmax > latest_input):
                            latest_input = wmax
                    if latest_input is not None and pd.notna(latest_input):
                        if latest_input > ends_ts[-1]:
                            ends_ts.append(pd.to_datetime(latest_input, utc=True))
                except Exception:
                    pass

                if ends_ts:
                    epoch_end_ts_list = sorted(ends_ts)
                    burn_dates_are_ends = True
        except Exception:
            epoch_end_ts_list = []

    # ...existing code...
    # If burns.json provided canonical epoch end timestamps, apply known one-day shifts
    # for problematic epochs so transactions fall into the intended epoch.
    if epoch_end_ts_list:
        # 1-based epoch numbers that need shifting forward one day
        _shift_forward_epochs = {6, 21, 39, 40, 42, 43, 44, 48, 49, 63, 71}
        shifted_any = False
        print(f"[nodeProcessing] burns.json provided {len(epoch_end_ts_list)} epoch end timestamps (will apply shifts for {sorted(_shift_forward_epochs)})")
        for idx, ts in enumerate(epoch_end_ts_list):
            epoch_num = idx + 1
            try:
                orig_ts = pd.to_datetime(ts, utc=True)
            except Exception:
                orig_ts = ts
            if epoch_num in _shift_forward_epochs:
                # ensure ts is a pandas Timestamp and add one day (keeps timezone)
                try:
                    new_ts = pd.to_datetime(orig_ts, utc=True) + pd.Timedelta(days=1)
                    epoch_end_ts_list[idx] = new_ts
                    print(f"[nodeProcessing] SHIFTED FORWARD epoch {epoch_num}: {orig_ts.isoformat()} -> {new_ts.isoformat()}")
                    shifted_any = True
                except Exception as e:
                    print(f"[nodeProcessing] failed to shift epoch {epoch_num} ({e}); leaving as {orig_ts}")
            else:
                try:
                    print(f"[nodeProcessing] epoch {epoch_num}: {orig_ts.isoformat()} (no shift)")
                except Exception:
                    print(f"[nodeProcessing] epoch {epoch_num}: {orig_ts} (no shift)")
        if not shifted_any:
            print("[nodeProcessing] No epoch end timestamps required shifting.")
        # final summary
        try:
            final_list = [pd.to_datetime(x, utc=True).isoformat() for x in epoch_end_ts_list]
            print(f"[nodeProcessing] Final epoch_end_ts_list (first 10 shown): {final_list[:10]}")
        except Exception:
            print("[nodeProcessing] Final epoch_end_ts_list prepared.")
    # ...existing code...

    # Fallbacks if no burns.json end timestamps available:
    if not epoch_end_ts_list:
        # Use union of avail/work dates (normalized), then treat them as increasing "ends"
        dates_av = set(n_av["epoch_date"]) if not n_av.empty else set()
        dates_work = set(n_work["epoch_date"]) if not n_work.empty else set()
        all_dates = sorted(
            d for d in (dates_av | dates_work)
            if pd.notna(pd.to_datetime(d, errors="coerce"))
        )
        epoch_end_ts_list = [pd.to_datetime(d, utc=True) for d in all_dates]
        burn_dates_are_ends = True  # we treat these as epoch ends

    # --- Back-compat shim for the rest of aggregate_node_rewards ---
    # Convert end timestamps into ISO 8601 strings (keep time-of-day + 'Z')
    # so later code that does `elif burn_dates:` still works.
    burn_dates = [ts.isoformat().replace('+00:00', 'Z') for ts in epoch_end_ts_list] if epoch_end_ts_list else []
    # We are providing canonical *end* instants, not starts:
    burn_dates_are_ends = bool(epoch_end_ts_list)

    # Build epoch maps from the UTC end timestamps
    date_to_epoch = {ts.isoformat(): i + 1 for i, ts in enumerate(epoch_end_ts_list)}
    epoch_to_date = {i + 1: ts.isoformat() for i, ts in enumerate(epoch_end_ts_list)}

    # Save a reference CSV with full timestamps (not midnight-truncated)
    try:
        epoch_map_df = pd.DataFrame(
            {"epoch": list(range(1, len(epoch_end_ts_list) + 1)),
            "date": [ts.isoformat() for ts in epoch_end_ts_list]}
        )
        epoch_map_path = output_csv.parent.joinpath("epoch_map.csv")
        epoch_map_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_map_df.to_csv(epoch_map_path, index=False)
    except Exception:
        pass


    # Prefer burn end dates from burns.json when available. If burns.json
    # provides canonical epoch end dates we should ignore any existing
    # epoch_map.csv so the map is always generated from the authoritative source.
    project_root = Path(__file__).resolve().parent.parent
    epoch_map_path = project_root / "data" / "epoch_map.csv"
    epoch_map_from_file = None
    # If burn end dates are available, skip loading epoch_map.csv to avoid
    # mismatches (we will generate/overwrite the map from burns.json below).
    if epoch_map_path.exists() and not burn_dates_are_ends:
        try:
            emf = pd.read_csv(epoch_map_path)
            # Expect columns: epoch, date (ISO date strings)
            # The date column contains epoch START dates (when each epoch begins)
            emf = emf.dropna(subset=["date"]).copy()
            emf["date"] = pd.to_datetime(emf["date"], errors="coerce", utc=True)
            emf = emf.sort_values(by=["epoch"] if "epoch" in emf.columns else ["date"]).reset_index(drop=True)
            # keep parsed Timestamps and integer epoch numbers
            emf = emf.assign(epoch=emf["epoch"].astype(int) if "epoch" in emf.columns else range(1, len(emf) + 1))
            
            # Convert epoch START dates to END timestamps for use in searchsorted
            # Each epoch ends just before the next epoch starts
            # For the last epoch, use a far-future date as the end
            epoch_end_ts_list = []
            for i in range(len(emf)):
                if i < len(emf) - 1:
                    # End is just before next epoch start (next epoch's start timestamp)
                    epoch_end_ts_list.append(emf.iloc[i + 1]["date"])
                else:
                    # Last epoch: use far future
                    epoch_end_ts_list.append(pd.Timestamp('2099-12-31', tz='UTC'))
            
            epoch_map_from_file = emf[["epoch", "date"]]
            # Build mappings using end timestamps
            date_to_epoch = {ts.isoformat(): int(e) for e, ts in zip(emf["epoch"].astype(int), epoch_end_ts_list)}
            epoch_to_date = {int(e): ts.isoformat() for e, ts in zip(emf["epoch"].astype(int), epoch_end_ts_list)}
            unique_list = list(date_to_epoch.keys())
            burn_dates_are_ends = True  # We've converted starts to ends
        except Exception:
            epoch_map_from_file = None
            unique_list = []
    elif burn_dates:
        unique_list = burn_dates
    else:
        dates_av = set(n_av["epoch_date"]) if not n_av.empty else set()
        dates_work = set(n_work["epoch_date"]) if not n_work.empty else set()
        all_dates = dates_av.union(dates_work)
        # parse and sort chronologically, keep only valid dates and normalize to ISO date strings
        parsed = []
        for d in all_dates:
            dt = pd.to_datetime(d, errors="coerce")
            if pd.notna(dt):
                nd = dt.normalize().date().isoformat()
                parsed.append((nd, dt.normalize()))
        # sort by datetime and preserve normalized date strings, dedupe while preserving order
        parsed_sorted = sorted(parsed, key=lambda x: x[1])
        unique_list = list(dict.fromkeys([nd for nd, _ in parsed_sorted]))

    if not unique_list:
        # no valid dates found, produce empty result
        res = pd.DataFrame(columns=["recipient_wallet", "first_epoch"]) 
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(output_csv, index=False)
        return res

    # If we already have canonical epoch end dates (from epoch_map.csv or
    # from burns.json endDate), use those directly. Otherwise, unique_list
    # currently contains epoch start dates inferred from inputs, so compute
    # inclusive epoch end dates as the day before the next epoch's start.
    if epoch_map_from_file is not None:
        # epoch_map_from_file already set date_to_epoch/epoch_to_date earlier
        pass
    elif burn_dates_are_ends:
        # burn_dates already represent epoch end dates in chronological order
        date_to_epoch = {d: i + 1 for i, d in enumerate(unique_list)}
        epoch_to_date = {i + 1: d for i, d in enumerate(unique_list)}
    else:
        date_to_epoch = {d: i + 1 for i, d in enumerate(unique_list)}
        # compute canonical epoch end dates (inclusive): each epoch ends the day
        # before the next epoch's start. For the last epoch we use its start date
        # as the end date (no later information available).
        starts = list(pd.to_datetime(unique_list).normalize())
        # compute inclusive end date for each epoch as the day before the next epoch's start
        ends = []
        for i, s in enumerate(starts):
            if i + 1 < len(starts):
                end_dt = (starts[i + 1] - pd.Timedelta(days=1)).to_pydatetime()
            else:
                # last epoch: use its start as the end (no later info)
                end_dt = s.to_pydatetime()
            ends.append(end_dt)
        epoch_to_date = {i + 1: ends[i].date().isoformat() for i in range(len(ends))}
    try:
        epoch_map_df = pd.DataFrame(list(epoch_to_date.items()), columns=["epoch", "date"])
        epoch_map_path = output_csv.parent.joinpath("epoch_map.csv")
        epoch_map_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_map_df.to_csv(epoch_map_path, index=False)
    except Exception:
        # non-fatal: continue without writing the map
        pass

    # assign epoch numbers to normalized frames. If an epoch_map.csv file was
    # found above we use its Timestamp starts to bucket by half-open intervals
    # [start_i, start_{i+1}) so datetimes between epoch starts are assigned to
    # the earlier epoch. Otherwise we map by normalized date strings.
    def _assign_epoch(df: pd.DataFrame) -> pd.DataFrame:
        """Assign numeric epoch numbers to rows using canonical UTC end timestamps.

        Rules:
        - If a numeric 'epoch' column exists, trust it.
        - Else, map epoch_dt to the first epoch whose *end* >= (epoch_dt - grace).
        - A small post-epoch grace (EPOCH_GRACE_HOURS) keeps just-after-boundary
            payouts in their intended epoch.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["recipient_wallet", "epoch_date", "rewards", "epoch"])

        df = df.copy()
        df["recipient_wallet"] = df["recipient_wallet"].astype(str).str.lower().str.strip()

                #date/blocktime) in input dataframe")
        
        df = df.copy()
        df["recipient_wallet"] = df["recipient_wallet"].astype(str).str.lower().str.strip()

        # 1) Trust explicit epoch ids when present, but only if they look reasonable
        # Reasonable epoch numbers are small positive integers (1-200 range expected)
        # Reject Unix timestamps (>1000000) and obviously wrong values (<=0, or 1 for recent dates)
        if "epoch" in df.columns:
            try:
                epn = pd.to_numeric(df["epoch"], errors="coerce")
                # Filter to reasonable epoch values: positive integers < 1000
                reasonable_mask = (epn > 0) & (epn < 1000) & (epn == epn.round())
                if reasonable_mask.sum() > 0:
                    # Further validation: if we have dates, check if epoch 1 makes sense
                    # Epoch 1 should only be for dates in late 2023/early 2024
                    if "epoch_dt" in df.columns:
                        dt_series = pd.to_datetime(df["epoch_dt"], errors="coerce", utc=True)
                        # Epoch 1 ended around 2024-01-03, so reject epoch=1 for dates after 2024-02-01
                        epoch1_wrong = (epn == 1) & (dt_series > pd.Timestamp('2024-02-01', tz='UTC'))
                        reasonable_mask = reasonable_mask & ~epoch1_wrong
                    
                    if reasonable_mask.sum() > 0:
                        # Store rows with valid epochs for later concatenation
                        df_with_valid_epochs = df[reasonable_mask].copy()
                        df_with_valid_epochs["epoch"] = epn[reasonable_mask].astype(int)
                        # Continue processing rows without valid epochs
                        df = df[~reasonable_mask].copy()
                        has_valid_epochs = True
                    else:
                        # All epoch values were unreasonable, ignore them
                        has_valid_epochs = False
                else:
                    # No reasonable epoch values, fall through to timestamp logic
                    has_valid_epochs = False
            except Exception:
                has_valid_epochs = False
        else:
            has_valid_epochs = False
        
        # Store the valid epochs dataframe if we have one
        df_with_valid_epochs = df_with_valid_epochs if has_valid_epochs else None

        # 2) Use timestamps (UTC)
        # Note: when using canonical epoch end dates from burns.json, we don't apply
        # the grace period since those dates are already authoritative
        if "epoch_dt" not in df.columns:
            df["epoch_dt"] = pd.to_datetime(df["epoch_date"], errors="coerce", utc=True)
        else:
            df["epoch_dt"] = pd.to_datetime(df["epoch_dt"], errors="coerce", utc=True)
        df["epoch_date"] = df["epoch_dt"].dt.normalize().dt.date.astype(str)
        # canonical ends (UTC) from the block above
        # Use pandas for timezone-aware handling and convert to numpy datetime64[ns]
        # which has no tzinfo. This avoids the "no explicit representation of
        # timezones" UserWarning and safely handles NaT values present in
        # `df["epoch_dt"]` (which may be floats/NaN).
        try:
            ends_pd = pd.to_datetime(list(epoch_end_ts_list), utc=True, errors="coerce")
            # remove tzinfo if present to get timezone-naive numpy datetimes
            try:
                if getattr(ends_pd.dt, 'tz', None) is not None:
                    ends_pd = ends_pd.dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception:
                # if ends_pd isn't a Series/Index with .dt, ignore
                pass
            ends_ts = ends_pd.to_numpy(dtype='datetime64[ns]')
        except Exception:
            # Fallback: empty array
            ends_ts = np.array([], dtype='datetime64[ns]')

        # Convert the dataframe epoch_dt column to datetime64[ns] safely
        dt_series = pd.to_datetime(df.get("epoch_dt", pd.Series([], dtype=object)), utc=True, errors="coerce")
        # If tz-aware, convert to UTC then drop tzinfo to get naive datetimes
        try:
            if getattr(dt_series.dt, 'tz', None) is not None:
                dt_series = dt_series.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            # if dt accessor isn't available or other error, continue
            pass
        dt_vals = dt_series.to_numpy(dtype='datetime64[ns]')

        # Map each valid dt to the epoch whose end date is >= dt
        # ends_ts[i] represents the end date of epoch i+1 (0-based indexing)
        # Use searchsorted to find which epoch each date belongs to
        # Handle NaT/invalid datetimes by leaving epoch as -1
        epoch_arr = np.full(len(df), -1, dtype=int)
        if len(ends_ts) > 0 and len(dt_vals) > 0:
            valid_mask = ~pd.isna(dt_series)
            if valid_mask.any():
                # searchsorted finds the index where dt would be inserted
                # With side='left', we find the first epoch end >= the date
                # Since ends_ts uses 0-based indexing (ends_ts[0] = end of epoch 1),
                # idx+1 gives us the 1-based epoch number
                # Example: dt=2024-01-17 00:00, ends_ts[3]=2024-01-17 02:00 (epoch 4 end)
                #   searchsorted returns idx=3, epoch should be 3+1=4
                idx = np.searchsorted(ends_ts, dt_vals[valid_mask], side='left')
                # Convert 0-based index to 1-based epoch number
                epoch_arr_valid = idx + 1
                # Clamp to valid range [1, len(ends_ts)]
                epoch_arr_valid = np.clip(epoch_arr_valid, 1, len(ends_ts))
                epoch_arr[valid_mask] = epoch_arr_valid

        df["epoch"] = pd.Series([int(x) if x > 0 else pd.NA for x in epoch_arr], index=df.index)
        df = df.dropna(subset=["epoch"])
        if not df.empty:
            df["epoch"] = df["epoch"].astype(int)
        
        # Combine with rows that had valid pre-existing epochs
        if df_with_valid_epochs is not None:
            df = pd.concat([df_with_valid_epochs, df], ignore_index=True)
        
        if df.empty:
            return pd.DataFrame(columns=["recipient_wallet", "epoch_date", "rewards", "epoch"])
        
        return df


    n_av = _assign_epoch(n_av)
    n_work = _assign_epoch(n_work)

    # recompute combined for first_epoch calculation (now with numeric epoch)
    combined_for_first = pd.concat([n_av[['recipient_wallet', 'epoch']], n_work[['recipient_wallet', 'epoch']]], ignore_index=True)
    first_epoch = combined_for_first.groupby('recipient_wallet')['epoch'].min().rename('first_epoch')

    # pivot per-epoch numeric columns for avail and work separately
    def _pivot_by_epoch(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        pv = df.groupby(['recipient_wallet', 'epoch'], as_index=False)['rewards'].sum()
        pv = pv.pivot_table(index='recipient_wallet', columns='epoch', values='rewards', fill_value=0.0)
        pv.columns = [f"{prefix}E{int(c)}" for c in pv.columns]
        return pv

    pv_av = _pivot_by_epoch(n_av, 'avail')
    pv_work = _pivot_by_epoch(n_work, 'work')

    # merge the two pivots (outer join on recipient_wallet index)
    if pv_av.empty and pv_work.empty:
        merged = pd.DataFrame()
    elif pv_av.empty:
        merged = pv_work.copy()
    elif pv_work.empty:
        merged = pv_av.copy()
    else:
        merged = pv_av.join(pv_work, how='outer').fillna(0.0)

    merged = merged.fillna(0.0)

    # assemble final DataFrame: recipient_wallet, first_epoch, then epoch columns ordered by epoch number
    result = merged.reset_index()
    result = result.merge(first_epoch.reset_index(), on='recipient_wallet', how='left')

    # include a JSON column mapping epoch numbers to ISO date strings for easy reference
    try:
        epoch_dates_json = json.dumps(epoch_to_date)
        result["epoch_dates"] = epoch_dates_json
    except Exception:
        result["epoch_dates"] = ""

    # normalize recipient_wallet strings and collapse duplicates if any
    result['recipient_wallet'] = result['recipient_wallet'].astype(str).str.lower().str.strip()

    # if accidental duplicate recipient_wallet rows exist (index type mismatches etc.),
    # group by recipient_wallet: sum epoch columns, take min(first_epoch)
    epoch_cols = [c for c in result.columns if c not in ('recipient_wallet', 'first_epoch')]
    if result['recipient_wallet'].duplicated().any():
        agg_map = {c: 'sum' for c in epoch_cols}
        agg_map['first_epoch'] = 'min'
        result = result.groupby('recipient_wallet', as_index=False).agg(agg_map)
    else:
        epoch_cols = [c for c in result.columns if c not in ('recipient_wallet', 'first_epoch')]
    # filter to node set (only wallets present in avail dataset)
    if node_set:
        result = result[result['recipient_wallet'].isin(node_set)]
    import re

    def _epoch_num_from_col(col: str):
        m = re.search(r'E(\d+)$', col)
        return int(m.group(1)) if m else float('inf')

    epoch_cols_sorted = sorted(epoch_cols, key=lambda c: (_epoch_num_from_col(c), c))
    cols = ['recipient_wallet', 'first_epoch'] + epoch_cols_sorted
    cols = [c for c in cols if c in result.columns]
    result = result[cols]

    # Calculate ratio, SMA 12, and Tier columns for each epoch per wallet
    # Compute a full range of epochs from 1 to the maximum epoch present in avail or work columns
    # Determine max epoch using the canonical epoch map if available, otherwise
    # fall back to existing avail/work columns. This ensures we include epochs
    # even if no wallet had rewards that epoch (we will create zero columns).
    try:
        if isinstance(epoch_to_date, dict) and epoch_to_date:
            max_ep = max(int(k) for k in epoch_to_date.keys())
        else:
            raise NameError
    except Exception:
        avail_ep = [int(col.replace('availE', '')) for col in result.columns if col.startswith('availE')]
        work_ep = [int(col.replace('workE', '')) for col in result.columns if col.startswith('workE')]
        if avail_ep or work_ep:
            max_ep = max(avail_ep + work_ep)
        else:
            max_ep = 0
    ep_nums = list(range(1, max_ep + 1))

    # Ensure the DataFrame has explicit availE/workE columns for every epoch
    # in the canonical range. Missing columns are added with 0.0 so downstream
    # logic can rely on a complete epoch sequence.
    for ep in ep_nums:
        a_col = f"availE{ep}"
        w_col = f"workE{ep}"
        if a_col not in result.columns:
            result[a_col] = 0.0
        if w_col not in result.columns:
            result[w_col] = 0.0

    # Reorder epoch columns consistently: interleave avail/work by epoch number
    epoch_pairs = []
    for ep in ep_nums:
        epoch_pairs.extend([f"availE{ep}", f"workE{ep}"])
    # keep only columns that exist in result (recipient_wallet/first_epoch + epoch pairs)
    cols = [c for c in ['recipient_wallet', 'first_epoch'] + epoch_pairs if c in result.columns]
    # preserve any other columns after the epoch block
    remaining = [c for c in result.columns if c not in cols]
    result = result[cols + remaining]

    # Calculate ratio, SMA 12, and Tier columns for each epoch per wallet
    # Compute a full range of epochs from 1 to the maximum epoch present in avail or work columns
    avail_ep = [int(col.replace('availE', '')) for col in result.columns if col.startswith('availE')]
    work_ep = [int(col.replace('workE', '')) for col in result.columns if col.startswith('workE')]
    if avail_ep or work_ep:
        max_ep = max(avail_ep + work_ep)
    else:
        max_ep = 0
    ep_nums = list(range(1, max_ep + 1))
    
    def compute_metrics(row):
        # Compute simple ratio for each epoch: work/avail if avail > 0, else NaN
        ratios = []
        for ep in ep_nums:
            avail = row.get(f'availE{ep}', 0)
            work = row.get(f'workE{ep}', 0)
            r = work / avail if avail > 0 else np.nan
            ratios.append(r)
        
        # Find the first epoch index where a valid ratio exists
        first_valid = None
        for i, r in enumerate(ratios):
            if not np.isnan(r):
                first_valid = i
                break
        
        # If a valid ratio exists, backfill all earlier epochs with the first valid value
        if first_valid is not None:
            for i in range(first_valid):
                ratios[i] = ratios[first_valid]
        
        # Compute SMA 12 for each epoch as the rolling average of up to 12 ratios
        sma = []
        for i in range(len(ratios)):
            window = ratios[max(0, i - 11):(i + 1)]
            valid_window = [v for v in window if not np.isnan(v)]
            s = np.mean(valid_window) if valid_window else np.nan
            sma.append(s)
        
        # Backfill SMA before the first valid ratio
        if first_valid is not None:
            for i in range(first_valid):
                sma[i] = sma[first_valid]
        
        # Compute tier for each epoch: use SMA if available, otherwise ratio
        tiers = []
        for r, s in zip(ratios, sma):
            use_val = s if not np.isnan(s) else r
            if np.isnan(use_val):
                tiers.append(np.nan)
            else:
                tiers.append('T2' if use_val > 4 else 'T3')
        
        return pd.Series(ratios + sma + tiers)
    
    metric_cols = [f'ratioE{ep}' for ep in ep_nums] + [f'smaE{ep}' for ep in ep_nums] + [f'tierE{ep}' for ep in ep_nums]
    metrics_df = result.apply(compute_metrics, axis=1)
    metrics_df.columns = metric_cols
    result = pd.concat([result, metrics_df], axis=1)

    # Cull wallets that only appear in a small set of epochs
    # If a wallet has non-zero avail/work only in epochs {3,11,16,22,27} and
    # no other epochs, remove it from the result and report how many were culled.
    try:
        special_epochs = {2,3,6,11,16,22, 23, 27}
        # ep_nums is the canonical list of epoch numbers computed earlier
        # Build a boolean mask of rows to cull
        def _present_epochs_set(row):
            s = set()
            for ep in ep_nums:
                a = row.get(f"availE{ep}", 0)
                w = row.get(f"workE{ep}", 0)
                try:
                    a_val = float(a) if not pd.isna(a) else 0.0
                except Exception:
                    a_val = 0.0
                try:
                    w_val = float(w) if not pd.isna(w) else 0.0
                except Exception:
                    w_val = 0.0
                if (a_val + w_val) > 0:
                    s.add(ep)
            return s

        present_sets = result.apply(_present_epochs_set, axis=1)
        to_cull_mask = present_sets.apply(lambda s: bool(s) and s.issubset(special_epochs))
        culled_count = int(to_cull_mask.sum())
        if culled_count > 0:
            print(f"Culled {culled_count} addresses that only appeared in epochs {sorted(list(special_epochs))}")
            result = result.loc[~to_cull_mask].reset_index(drop=True)
    except Exception:
        # non-fatal: if the culling logic fails for any reason, skip it
        pass

    # write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    return result


def plot_sma_by_tier(use_dates=False):
    """Plot the 12-epoch SMA of counts of wallets receiving work (overall and by tier).

    For each epoch present as a `workE{n}` column, compute counts of wallets
    with `workE{n}` > 0 (this yields a value between 0 and the number of
    unique wallets, e.g., 294). Also compute counts restricted to wallets
    labeled 'T2' and 'T3' for the same epoch. Then compute a 12-epoch SMA
    of each counts series and plot them on a dark background using pastel
    colors.
    
    Parameters
    ----------
    use_dates : bool
        If True, use month/year dates on X-axis (requires epoch_date mapping). If False, use epoch numbers.
    """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Prefer culled node summary if present
    culled = project_root / "data" / "node_summary_culled.csv"
    summary_path = culled if culled.exists() else project_root / "data" / "node_summary.csv"
    if not summary_path.exists():
        print("Node summary CSV not found.")
        return
    print(f"Using summary file: {summary_path}")
    df = pd.read_csv(summary_path)

    # Discover epoch numbers from work columns
    work_cols = [c for c in df.columns if c.startswith("workE")]
    if not work_cols:
        print("No work columns found in summary.")
        return
    ep_nums = sorted([int(c.replace("workE", "")) for c in work_cols])

    counts_total = []
    counts_t2 = []
    counts_t3 = []

    for ep in ep_nums:
        work_col = f"workE{ep}"
        tier_col = f"tierE{ep}"

        # if work_col not in df.columns:
        #     # If the column is missing assume zero recipients for alignment
        #     counts_total.append(0)
        #     counts_t2.append(0)
        #     counts_t3.append(0)
        #     continue

        work_series = pd.to_numeric(df[work_col], errors="coerce").fillna(0.0)
        has_work = work_series > 0
        counts_total.append(int(has_work.sum()))

        if tier_col in df.columns:
            tier_vals = df[tier_col].astype(str).fillna("")
            counts_t2.append(int((has_work & (tier_vals == 'T2')).sum()))
            counts_t3.append(int((has_work & (tier_vals == 'T3')).sum()))
        else:
            counts_t2.append(0)
            counts_t3.append(0)

    # Build Series and compute 12-epoch SMA (min_periods=1)
    s_tot = pd.Series(counts_total, index=ep_nums, dtype=float)
    s_t2 = pd.Series(counts_t2, index=ep_nums, dtype=float)
    s_t3 = pd.Series(counts_t3, index=ep_nums, dtype=float)

    # Compute trailing 12-epoch SMA using pandas rolling (looks at current and past epochs).
    # Requirement: exclude epoch 1 from SMA calculations so the windows start at epoch 2.
    # Achieve this by creating copies and forcing index 1 -> NaN before rolling.
    s_tot_for_roll = s_tot.copy()
    s_t2_for_roll = s_t2.copy()
    s_t3_for_roll = s_t3.copy()
    if 1 in s_tot_for_roll.index:
        s_tot_for_roll.loc[1] = np.nan
        s_t2_for_roll.loc[1] = np.nan
        s_t3_for_roll.loc[1] = np.nan

    sma_tot = s_tot_for_roll.rolling(window=12, min_periods=1).mean()
    sma_t2 = s_t2_for_roll.rolling(window=12, min_periods=1).mean()
    sma_t3 = s_t3_for_roll.rolling(window=12, min_periods=1).mean()

    # Backfill any initial NaNs with the first valid SMA so the plotted lines
    # keep their initial level (behaves like the white backfilled lines in the ref).
    # Use .bfill() (preferred over deprecated fillna(method='bfill')).
    sma_tot = sma_tot.bfill().fillna(0.0)
    sma_t2 = sma_t2.bfill().fillna(0.0)
    sma_t3 = sma_t3.bfill().fillna(0.0)

    # Plot starting from epoch 1
    ep_plot = [ep for ep in ep_nums if ep >= 1]
    if not ep_plot:
        print("No epochs >= 1 to plot.")
        return

    sma_tot_plot = sma_tot.loc[ep_plot]
    sma_t2_plot = sma_t2.loc[ep_plot]
    sma_t3_plot = sma_t3.loc[ep_plot]

    # Determine x-axis: epochs or dates
    if use_dates:
        # Try to load epoch-to-date mapping from epoch_map.csv
        project_root = Path(__file__).resolve().parent.parent
        epoch_map_path = project_root / "data" / "epoch_map.csv"
        if epoch_map_path.exists():
            try:
                epoch_map = pd.read_csv(epoch_map_path)
                if "epoch" in epoch_map.columns and "date" in epoch_map.columns:
                    epoch_map["date"] = pd.to_datetime(epoch_map["date"], errors="coerce")
                    epoch_to_date = dict(zip(epoch_map["epoch"], epoch_map["date"]))
                    x_nums = [epoch_to_date.get(ep, ep) for ep in ep_nums]
                    x_plot = [epoch_to_date.get(ep, ep) for ep in ep_plot]
                    x_title = "Date"
                else:
                    print("Warning: epoch_map.csv missing required columns, using epochs")
                    x_nums = ep_nums
                    x_plot = ep_plot
                    x_title = "Epoch"
            except Exception as e:
                print(f"Warning: Could not load epoch_map.csv ({e}), using epochs")
                x_nums = ep_nums
                x_plot = ep_plot
                x_title = "Epoch"
        else:
            print("Warning: epoch_map.csv not found, using epochs")
            x_nums = ep_nums
            x_plot = ep_plot
            x_title = "Epoch"
    else:
        x_nums = ep_nums
        x_plot = ep_plot
        x_title = "Epoch"

    # Prefer interactive Plotly HTML output; fallback to matplotlib PNG if plotly missing
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # build figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # raw totals as transparent bars
        fig.add_trace(
            go.Bar(x=x_nums, y=s_tot.values, name='Raw total wallets', marker_color='#7f7f7f', opacity=0.25),
            secondary_y=False,
        )

        # SMA lines (no markers)
        fig.add_trace(go.Scatter(x=x_plot, y=sma_tot_plot.values, mode='lines', name='Total wallets (12 SMA)', line=dict(color='#aec7e8')),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=x_plot, y=sma_t2_plot.values, mode='lines', name='T2 wallets (12 SMA)', line=dict(color='#ffb3c1')),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=x_plot, y=sma_t3_plot.values, mode='lines', name='T3 wallets (12 SMA)', line=dict(color='#b5e1a5')),
                      secondary_y=False)

        # percent-change traces on secondary axis
        pct_tot = sma_tot.pct_change(periods=12) * 100
        pct_t2 = sma_t2.pct_change(periods=12) * 100
        pct_t3 = sma_t3.pct_change(periods=12) * 100
        pct_tot = pct_tot.replace([np.inf, -np.inf], np.nan)
        pct_t2 = pct_t2.replace([np.inf, -np.inf], np.nan)
        pct_t3 = pct_t3.replace([np.inf, -np.inf], np.nan)

        # Backfill first valid percent-change to epoch 2
        def _backfill_first_to_epoch2(s: pd.Series) -> pd.Series:
            if s.notna().sum() == 0:
                return s.fillna(0.0)
            fv = s.first_valid_index()
            if fv is None:
                return s.fillna(0.0)
            try:
                fv_int = int(fv)
            except Exception:
                return s.bfill().fillna(0.0)
            if fv_int > 2:
                fill_val = s.loc[fv_int]
                for idx in list(s.index):
                    try:
                        i = int(idx)
                    except Exception:
                        continue
                    if 2 <= i < fv_int:
                        s.loc[i] = fill_val
            return s.bfill().fillna(0.0)

        pct_tot = _backfill_first_to_epoch2(pct_tot)
        pct_t2 = _backfill_first_to_epoch2(pct_t2)
        pct_t3 = _backfill_first_to_epoch2(pct_t3)

        pct_tot_plot = pct_tot.loc[ep_plot]
        pct_t2_plot = pct_t2.loc[ep_plot]
        pct_t3_plot = pct_t3.loc[ep_plot]

        fig.add_trace(go.Scatter(x=x_plot, y=pct_tot_plot.values, mode='lines', name='Total SMA % change (12-ep)', line=dict(color='#aec7e8', dash='dash')),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=x_plot, y=pct_t2_plot.values, mode='lines', name='T2 SMA % change (12-ep)', line=dict(color='#ffb3c1', dash='dash')),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=x_plot, y=pct_t3_plot.values, mode='lines', name='T3 SMA % change (12-ep)', line=dict(color='#b5e1a5', dash='dash')),
                      secondary_y=True)

        # layout
        fig.update_layout(
            title={
                'text': 'Node Wallet Trends',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=x_title,
            margin=dict(r=10)  # Reduce right margin to eliminate blank space
        )
        # dark theme: black background and light axis/grid colors to match matplotlib dark style
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0.0)', font=dict(color='white'), orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.85),
        )
        fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False,
                         showline=True, linecolor='rgba(255,255,255,0.2)', tickfont=dict(color='white'), title_font=dict(color='white'))
        fig.update_yaxes(title_text='Wallets', secondary_y=False,
                         showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                         showline=True, linecolor='rgba(255,255,255,0.2)', tickfont=dict(color='white'), title_font=dict(color='white'))
        fig.update_yaxes(title_text='12 Epoch % Change', secondary_y=True,
                         showgrid=False, zeroline=False, showline=True, linecolor='rgba(255,255,255,0.2)',
                         tickfont=dict(color='white'), title_font=dict(color='white'))

        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir.joinpath('node_sma_chart.html')
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        print(f"Interactive chart saved to {out_path}")
        return
    except Exception:
        # fallback to matplotlib PNG if plotly not available or any plotting error
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(ep_nums, s_tot.values, color='#7f7f7f', alpha=0.25, label='Raw total wallets', zorder=1)
        ax.plot(ep_plot, sma_tot_plot.values, color='#aec7e8', zorder=3, label='Total wallets (12 SMA)')
        ax.plot(ep_plot, sma_t2_plot.values, color='#ffb3c1', zorder=4, label='T2 wallets (12 SMA)')
        ax.plot(ep_plot, sma_t3_plot.values, color='#b5e1a5', zorder=4, label='T3 wallets (12 SMA)')
        ax2 = ax.twinx()
        pct_tot = sma_tot.pct_change(periods=12) * 100
        pct_t2 = sma_t2.pct_change(periods=12) * 100
        pct_t3 = sma_t3.pct_change(periods=12) * 100
        pct_tot = pct_tot.replace([np.inf, -np.inf], np.nan)
        pct_t2 = pct_t2.replace([np.inf, -np.inf], np.nan)
        pct_t3 = pct_t3.replace([np.inf, -np.inf], np.nan)
        pct_tot = pct_tot.bfill().fillna(0.0)
        pct_t2 = pct_t2.bfill().fillna(0.0)
        pct_t3 = pct_t3.bfill().fillna(0.0)
        pct_tot_plot = pct_tot.loc[ep_plot]
        pct_t2_plot = pct_t2.loc[ep_plot]
        pct_t3_plot = pct_t3.loc[ep_plot]
        ax2.plot(ep_plot, pct_tot_plot.values, color='#aec7e8', linestyle='--', zorder=5, label='Total SMA % change (12-ep)')
        ax2.plot(ep_plot, pct_t2_plot.values, color='#ffb3c1', linestyle='--', zorder=5, label='T2 SMA % change (12-ep)')
        ax2.plot(ep_plot, pct_t3_plot.values, color='#b5e1a5', linestyle='--', zorder=5, label='T3 SMA % change (12-ep)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('12-Epoch SMA of wallets receiving work')
        ax2.set_ylabel('12-Epoch % change of SMA')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        fig.tight_layout()
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir.joinpath('node_sma_chart.png')
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Chart saved to {out_path} (matplotlib fallback)")


# Append plotting function call in main block
if __name__ == "__main__":
    # convenience CLI: run with default paths
    out = aggregate_node_rewards()
    print(f"Wrote node summary with {len(out)} wallets to data/node_summary.csv")
    # Generate the SMA chart by tier
    plot_sma_by_tier()

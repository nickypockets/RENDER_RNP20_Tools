import os
import json
import pandas as pd
import requests
from dune_client.client import DuneClient
from dotenv import load_dotenv
from pathlib import Path


# Determine project root (parent of tools directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load IDs from JSON config
CONFIG_PATH = PROJECT_ROOT / "settings" / "download_ids.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    ID = json.load(f)

# load .env
load_dotenv()

DUNE_API_KEY = os.environ.get("DUNE_API_KEY")
if not DUNE_API_KEY:
    raise RuntimeError("DUNE_API_KEY not set in environment (.env at project root).")
dune = DuneClient(DUNE_API_KEY)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _extract_rows_from_result(obj):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.get("rows", []) or obj.get("data", []) or []
    rows = getattr(obj, "rows", None)
    if rows is not None:
        return rows
    res = getattr(obj, "result", None)
    if res is not None and res is not obj:
        return _extract_rows_from_result(res)
    for attr in ("data", "rows", "result"):
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)):
            return val
    return []

def _ensure_epoch_end(df):
    # case-insensitive lookup
    cols = {c.lower(): c for c in df.columns}
    if "epoch_end" in cols:
        return df
    # Handle MM/DD/YYYY style in Date / date
    if "date" in cols:
        col = cols["date"]
        dt = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(df[col], errors="coerce")
        df["epoch_end"] = (dt.astype("int64") // 10**9).astype("float").astype("Int64")
        return df
    # Handle blocktime like "2025-09-01 23:40:18.000 UTC"
    if "blocktime" in cols:
        col = cols["blocktime"]
        s = df[col].astype(str).str.replace(r"\s*UTC$", "", regex=True)
        dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(s, utc=True, errors="coerce")
        df["epoch_end"] = (dt.astype("int64") // 10**9).astype("float").astype("Int64")
        return df
    return df

def download_all():
    """Download configured datasets. URL-configured IDs are saved as JSON files.

    This function is not executed on import to keep the module import-safe for tests.
    """
    keys_without_updates: list[str] = []
    keys_with_updates: list[str] = []
    for key, query_id in ID.items():
        folder_name = f"{key}_data"
        folder_path = DATA_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # If the configured ID is a URL, fetch JSON and save as JSON file directly.
        if isinstance(query_id, str) and query_id.lower().startswith("http"):
            try:
                resp = requests.get(query_id, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"Failed to fetch URL {query_id} for key {key}: {e}")
                continue

            out_path = folder_path / f"{key}.json"

            # Normalize incoming records: prefer dict['data'] if present
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                new_records = data.get("data")
                wrapper = True
            elif isinstance(data, list):
                new_records = data
                wrapper = False
            else:
                # unknown structure: write raw and continue
                try:
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, ensure_ascii=False, indent=2)
                    print(f"Saved JSON for key {key} -> {out_path} (unstructured payload)")
                except Exception as e:
                    print(f"Failed to write JSON for key {key}: {e}")
                continue

            # If file exists, merge by unique key (prefer 'id', fall back to 'startDate')
            added = 0
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                except Exception as e:
                    print(f"Failed to read existing JSON {out_path}: {e}")
                    existing = None

                if isinstance(existing, dict) and isinstance(existing.get("data"), list):
                    existing_records = existing.get("data")
                    existing_wrapper = True
                elif isinstance(existing, list):
                    existing_records = existing
                    existing_wrapper = False
                else:
                    # cannot interpret existing file; overwrite
                    existing_records = []
                    existing_wrapper = wrapper

                # determine key extractor
                def _key_for(rec):
                    if isinstance(rec, dict):
                        if "id" in rec:
                            return ("id", rec.get("id"))
                        if "startDate" in rec:
                            return ("startDate", rec.get("startDate"))
                    # fallback: tuple of items (not ideal but deterministic)
                    try:
                        return ("row", json.dumps(rec, sort_keys=True))
                    except Exception:
                        return ("row", str(rec))

                existing_keys = set(_key_for(r) for r in existing_records)
                to_add = []
                for rec in new_records:
                    k = _key_for(rec)
                    if k not in existing_keys:
                        to_add.append(rec)
                        existing_keys.add(k)

                if not to_add:
                    print(f"No new epochs for key {key}; {out_path} unchanged.")
                    keys_without_updates.append(key)
                    continue

                # Merge and write back preserving wrapper type if possible
                merged_records = existing_records + to_add
                try:
                    if existing_wrapper:
                        existing["data"] = merged_records
                        with open(out_path, "w", encoding="utf-8") as fh:
                            json.dump(existing, fh, ensure_ascii=False, indent=2)
                    else:
                        with open(out_path, "w", encoding="utf-8") as fh:
                            json.dump(merged_records, fh, ensure_ascii=False, indent=2)
                    added = len(to_add)
                    print(f"Appended {added} new epochs for key {key} -> {out_path}")
                    if added > 0:
                        keys_with_updates.append(key)
                except Exception as e:
                    print(f"Failed to merge/write JSON for key {key}: {e}")
                    continue
            else:
                # write fresh file with same wrapper type as incoming
                try:
                    if wrapper:
                        out_obj = dict(data=new_records)
                    else:
                        out_obj = new_records
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(out_obj, fh, ensure_ascii=False, indent=2)
                    added = len(new_records)
                    print(f"Saved JSON for key {key} -> {out_path} ({added} epochs)")
                    if added > 0:
                        keys_with_updates.append(key)
                except Exception as e:
                    print(f"Failed to write JSON for key {key}: {e}")
            continue

        # Otherwise treat it as a Dune query id and use the Dune client.
        try:
            query_result = dune.get_latest_result(query_id)
        except Exception as e:
            print(f"Failed to fetch query {query_id} for key {key}: {e}")
            continue

        rows = _extract_rows_from_result(query_result)
        df_new = pd.DataFrame(rows)
        if df_new.empty:
            print(f"No rows returned for key {key} (query {query_id}).")
            keys_without_updates.append(key)
            continue

        df_new = _ensure_epoch_end(df_new)

        # Write or append by diffing rows (append only rows not present in existing CSV)
        target_csv = os.path.join(folder_path, f"{key}.csv")
        added = 0
        if os.path.exists(target_csv):
            try:
                df_existing = pd.read_csv(target_csv, dtype=object)

                existing_cols = list(df_existing.columns)
                new_cols = [c for c in df_new.columns if c not in existing_cols]
                all_cols = existing_cols + new_cols

                # Ensure same columns
                for c in all_cols:
                    if c not in df_existing.columns:
                        df_existing[c] = pd.NA
                    if c not in df_new.columns:
                        df_new[c] = pd.NA

                df_existing_aligned = df_existing[all_cols]
                df_new_aligned = df_new[all_cols]

                existing_keys = set(df_existing_aligned.fillna("").astype(str).apply(lambda r: tuple(r), axis=1))
                new_keys = df_new_aligned.fillna("").astype(str).apply(lambda r: tuple(r), axis=1)

                mask_new = ~new_keys.isin(existing_keys)
                df_to_add = df_new.loc[mask_new]

                if df_to_add.empty:
                    print(f"No new unique rows for key {key}.")
                    added = 0
                    keys_without_updates.append(key)
                else:
                    if set(df_to_add.columns) != set(existing_cols):
                        combined = pd.concat([df_existing, df_to_add[all_cols]], ignore_index=True)
                        combined.drop_duplicates(inplace=True)
                        combined.to_csv(target_csv, index=False)
                        added = len(combined) - len(df_existing)
                    else:
                        df_to_add.to_csv(target_csv, mode="a", index=False, header=False)
                        added = len(df_to_add)
            except PermissionError as pe:
                print(f"Permission error writing {target_csv}: {pe}")
                print("Close apps that lock the file or adjust permissions.")
                continue
            except Exception as e:
                print(f"Failed to append for {key}: {e}")
                # fallback: overwrite with new data only
                df_new.to_csv(target_csv, index=False)
                added = len(df_new)
        else:
            df_new.to_csv(target_csv, index=False)
            added = len(df_new)

        print(f"Saved {added} new rows for key {key} -> {target_csv}")
        if added == 0:
            keys_without_updates.append(key)
        else:
            keys_with_updates.append(key)

    if keys_with_updates:
        stale_only = sorted(set(k for k in keys_without_updates if k not in keys_with_updates))
        if stale_only:
            print("Warning: no new data appended for the following data sources: " + ", ".join(stale_only))
    elif keys_without_updates:
        unique_keys = sorted(set(keys_without_updates))
        raise RuntimeError(
            "No new data appended for any data source. Affected keys: " + ", ".join(unique_keys)
        )


# If you already have a constant/path, keep it; otherwise:
WORK_CSV = PROJECT_ROOT / "data" / "work.csv"

def _merge_and_write_work_csv(new_df: pd.DataFrame, work_csv_path: Path | str = WORK_CSV,
                              dedupe_keys=("Recipient Wallet", "Date", "$RENDER Rewards")):
    """
    Merge new_df into work.csv, compute/keep epoch_end, de-dupe on keys, and write back.
    Deduplicates on (Recipient Wallet, Date, $RENDER Rewards) to remove exact duplicates
    while preserving multiple legitimate transactions to the same wallet on the same date.
    Returns the updated DataFrame.
    """
    work_csv_path = Path(work_csv_path)
    work_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize inputs: new_df may be a DataFrame or a path/str to a CSV file.
    if isinstance(new_df, (str, Path)):
        new_df = pd.read_csv(new_df)
    elif not isinstance(new_df, pd.DataFrame):
        # attempt to coerce to DataFrame (e.g. list/dict rows)
        try:
            new_df = pd.DataFrame(new_df)
        except Exception:
            raise TypeError("new_df must be a pandas.DataFrame or a path to a CSV file")

    if work_csv_path.exists():
        current = pd.read_csv(work_csv_path)
    else:
        current = pd.DataFrame()

    # Ensure both have epoch_end
    from tools.downloadData import _ensure_epoch_end  # reuse your existing function
    current = _ensure_epoch_end(current.copy()) if not current.empty else current
    new_df = _ensure_epoch_end(new_df.copy()) if not new_df.empty else new_df

    # Union & de-dupe
    if current.empty:
        out = new_df.copy()
    elif new_df.empty:
        out = current.copy()
    else:
        out = pd.concat([current, new_df], ignore_index=True)
        # drop duplicates on keys (keep last so “new” rows win)
        out = out.drop_duplicates(subset=list(dedupe_keys), keep="last")

    # nice, deterministic ordering
    if "epoch_end" in out.columns:
        out = out.sort_values("epoch_end").reset_index(drop=True)

    out.to_csv(work_csv_path, index=False)
    return out


if __name__ == "__main__":
    download_all()
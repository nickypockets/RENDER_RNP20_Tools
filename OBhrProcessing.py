"""OBhrProcessing.py

Assumptions:
- Use the latest node_operator epochId present in the OBhrs.json file to pick the tier column
  (i.e. use column name `tierE{max_epoch}` from `data/node_summary.csv`).
- Wallet comparison is case-insensitive (we lowercase both sides).

Produces:
- CSV at data/OBhrs_data/OBhrs_epoch_tier_totals.csv with columns: epochId,T2_total,T3_total
- Prints the number of unique wallets from OBhrs.json that were not mapped to T2 or T3.
"""
import json
import csv
import os
from collections import defaultdict


def load_obhrs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _simplify_addr(a):
    # keep only alphanumeric chars and lowercase for robust matching
    return ''.join(ch for ch in (a or '').lower() if ch.isalnum())


def load_node_summary(csv_path, tier_col):
    """Return two mappings:
    - mapping: wallet_lower -> tier (value from tier_col) or None if missing
    - simplified_map: simplified_alnum_lower -> tier (fallback)
    """
    # return mapping of wallet_norm -> full row dict, and simplified -> row dict
    mapping = {}
    simplified = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wallet = row.get('recipient_wallet') or row.get('recipient_WALLET')
            if not wallet:
                continue
            wallet_norm = wallet.strip().lower()
            mapping[wallet_norm] = row
            simp = _simplify_addr(wallet_norm)
            if simp and simp not in simplified:
                simplified[simp] = row
    return mapping, simplified


def process(obhrs_path, node_summary_csv, out_csv_path):
    data = load_obhrs(obhrs_path)
    epochs = data.get('data', {}).get('epochs', [])

    # determine latest node_operator epochId (use full set to pick tier column)
    node_epochs_full = [e for e in epochs if e.get('channel') == 'node_operator']
    if not node_epochs_full:
        raise SystemExit('no node_operator epochs found in OBhrs.json')

    max_epoch = max(e.get('epochId', 0) for e in node_epochs_full)
    tier_col = f'tierE{max_epoch}'

    wallet_to_tier, wallet_to_tier_simp = load_node_summary(node_summary_csv, tier_col)

    # filter to start rows at epoch 76 (inclusive)
    node_epochs = [e for e in node_epochs_full if (e.get('epochId') or 0) >= 76]
    if not node_epochs:
        raise SystemExit('no node_operator epochs >= 76 found in OBhrs.json')

    # Accumulate per-epoch totals
    rows = []
    unfiltered_wallets = set()
    seen_wallets = set()
    # track unmapped wallets per epoch: { epochId: { wallet: obh_total } }
    unmapped_by_epoch = defaultdict(lambda: defaultdict(float))

    for epoch in sorted(node_epochs, key=lambda e: e.get('epochId')):
        epoch_id = epoch.get('epochId')
        meta = epoch.get('metadataBySolKey', {}) or {}
        t2_sum = 0
        t3_sum = 0
        epoch_all_total = 0
        epoch_unmapped_total = 0

        for wallet, meta_v in meta.items():
            wallet_norm = wallet.strip().lower()
            seen_wallets.add(wallet_norm)
            obh = meta_v.get('obhUsed', 0) or 0
            epoch_all_total += obh
            tier = None
            row = wallet_to_tier.get(wallet_norm)
            if row:
                # try epoch-specific tier first, then fallback to latest tier_col
                per_epoch_col = f'tierE{epoch_id}'
                tier = row.get(per_epoch_col) or row.get(tier_col)
                if tier is not None:
                    tier = tier.strip() if tier.strip() != '' else None
            if tier is None:
                # fallback using simplified alnum form
                simp = _simplify_addr(wallet_norm)
                row2 = wallet_to_tier_simp.get(simp)
                if row2:
                    tier = row2.get(per_epoch_col) or row2.get(tier_col)
                    if tier is not None:
                        tier = tier.strip() if tier.strip() != '' else None
            if tier == 'T2':
                t2_sum += obh
            elif tier == 'T3':
                t3_sum += obh
            else:
                unfiltered_wallets.add(wallet_norm)
                epoch_unmapped_total += obh
                unmapped_by_epoch[epoch_id][wallet_norm] += obh

        # Keep raw T2 and a scaled version (T2 * 0.5) per epoch
        t2_raw = t2_sum
        t2_scaled = t2_sum * 0.5
        our_sum = t2_raw + t3_sum
        match = (our_sum == epoch_all_total)

        rows.append({
            'epochId': epoch_id,
            'T2_raw': t2_raw,
            'T2_scaled': t2_scaled,
            'T3_total': t3_sum,
            'epoch_all_total': epoch_all_total,
            'epoch_unmapped_total': epoch_unmapped_total,
            'our_sum': our_sum,
            'match': match,
        })

    # compute rolling 12 SMA for each column (use cumulative average until 12 points available)
    def rolling_sma(values, window=12):
        out = []
        for i in range(len(values)):
            start = max(0, i - (window - 1))
            window_vals = values[start:i + 1]
            avg = sum(window_vals) / len(window_vals) if window_vals else 0
            out.append(avg)
        return out

    t2_raw_values = [r['T2_raw'] for r in rows]
    t2_scaled_values = [r['T2_scaled'] for r in rows]
    t3_values = [r['T3_total'] for r in rows]

    sma_t2_raw = rolling_sma(t2_raw_values, window=12)
    sma_t2_scaled = rolling_sma(t2_scaled_values, window=12)
    sma_t3 = rolling_sma(t3_values, window=12)

    # attach SMA values to rows
    for idx, r in enumerate(rows):
        r['sma_T2_raw'] = sma_t2_raw[idx]
        r['sma_T2_scaled'] = sma_t2_scaled[idx]
        r['sma_T3_total'] = sma_t3[idx]

    # compute all-time totals (based on scaled T2 and T3 totals) and percentages
    total_t2_scaled_all = sum(t2_scaled_values)
    total_t3_all = sum(t3_values)
    denom = total_t2_scaled_all + total_t3_all
    if denom > 0:
        t2_all_pct = (total_t2_scaled_all / denom) * 100.0
        t3_all_pct = (total_t3_all / denom) * 100.0
    else:
        t2_all_pct = 0.0
        t3_all_pct = 0.0

    # compute per-epoch percentages (based on scaled T2 and T3 for that epoch)
    perc_t2_epoch = []
    perc_t3_epoch = []
    for t2s, t3 in zip(t2_scaled_values, t3_values):
        denom_p = t2s + t3
        if denom_p > 0:
            perc_t2_epoch.append((t2s / denom_p) * 100.0)
            perc_t3_epoch.append((t3 / denom_p) * 100.0)
        else:
            perc_t2_epoch.append(0.0)
            perc_t3_epoch.append(0.0)

    # apply the hybrid rolling SMA to the per-epoch percent series
    sma_perc_t2 = rolling_sma(perc_t2_epoch, window=12)
    sma_perc_t3 = rolling_sma(perc_t3_epoch, window=12)

    # attach all-time pct and SMA-smoothed rolling pct to rows
    for idx, r in enumerate(rows):
        r['T2_alltime_pct'] = t2_all_pct
        r['T3_alltime_pct'] = t3_all_pct
        r['rolling_pct_T2'] = sma_perc_t2[idx]
        r['rolling_pct_T3'] = sma_perc_t3[idx]

    # compute rolling percent-change of the SMA series over 12 periods
    # If there are missing epochs in the sequence, backfill those gaps using the first
    # available SMA value for stable percent-change calculation.
    def filled_pct_change_by_epoch(epoch_ids, values, periods=12):
        # map epoch -> value (values correspond to epoch_ids order)
        epoch_to_val = {eid: v for eid, v in zip(epoch_ids, values)}
        if not epoch_ids:
            return {}

        min_e = min(epoch_ids)
        max_e = max(epoch_ids)

        # find first available (non-None) value to use for backfilling — may be zero per Option A
        first_val = None
        for v in values:
            if v is not None:
                first_val = v
                break

        # build filled series across the full epoch range
        filled = []
        full_epochs = list(range(min_e, max_e + 1))
        for eid in full_epochs:
            v = epoch_to_val.get(eid)
            if v is None:
                # backfill missing or None with first available; if no first_val, keep None
                v = first_val
            filled.append(v)

        # compute percent change over `periods` on the filled series
        pct = [None] * len(filled)
        for i in range(len(filled)):
            if i >= periods and filled[i - periods] is not None and filled[i - periods] > 0:
                try:
                    pct[i] = (filled[i] / filled[i - periods] - 1.0) * 100.0
                except Exception:
                    pct[i] = None
            else:
                pct[i] = None

        # map pct values back to epoch ids (only for epochs present in original rows)
        epoch_to_pct = {}
        for idx, eid in enumerate(full_epochs):
            if eid in epoch_to_val:
                epoch_to_pct[eid] = pct[idx]

        # If earlier epochs have None, backfill them with the first available pct value
        # (per user request: backfill based on the first available value)
        first_non_none = None
        for val in pct:
            if val is not None:
                first_non_none = val
                break
        if first_non_none is not None:
            for eid in full_epochs:
                if eid in epoch_to_val and epoch_to_pct.get(eid) is None:
                    epoch_to_pct[eid] = first_non_none

        return epoch_to_pct

    epoch_ids = [r['epochId'] for r in rows]
    sma_pctchg_t2_by_epoch = filled_pct_change_by_epoch(epoch_ids, sma_t2_scaled, periods=12)
    sma_pctchg_t3_by_epoch = filled_pct_change_by_epoch(epoch_ids, sma_t3, periods=12)

    # attach SMA percent-change values to rows (lookup by epoch)
    for idx, r in enumerate(rows):
        r['rolling_pctchg_sma_T2_scaled'] = sma_pctchg_t2_by_epoch.get(r['epochId'])
        r['rolling_pctchg_sma_T3'] = sma_pctchg_t3_by_epoch.get(r['epochId'])

    # create an interactive HTML chart of the series
    def create_obhrs_chart(rows, out_html_path=os.path.join(os.path.dirname(out_csv_path), '..', 'reports', 'obhrs_sma_chart.html'), use_dates=False, show_raw_lines=False):
        """Create an interactive Plotly chart for OBhrs data.
        
        Parameters
        ----------
        rows : list
            List of dictionaries containing epoch data
        out_html_path : str
            Path to save the HTML chart
        use_dates : bool
            If True, use month/year dates on X-axis (requires epoch_date field). If False, use epoch numbers.
        show_raw_lines : bool
            If True, show the raw T2_scaled and T3_total lines with markers. If False (default), hide them.
        """
        # x axis will be epochId or dates
        if use_dates and 'epoch_date' in rows[0]:
            x = [r.get('epoch_date') for r in rows]
            x_title = 'Date'
        else:
            x = [r['epochId'] for r in rows]
            x_title = 'Epoch'
        x = [r['epochId'] for r in rows]
        # import plotly lazily to avoid import-time dependency failures in environments
        try:
            import plotly.graph_objects as go
        except Exception:
            print('plotly not available; skipping interactive chart generation')
            return

        fig = go.Figure()

        # raw series (lines+markers to mimic burnProcessing style) - optional
        if show_raw_lines:
            fig.add_trace(go.Scatter(x=x, y=[r['T2_scaled'] for r in rows], mode='lines+markers', name='T2_scaled', line=dict(color='#A5C8FF'), marker=dict(color='#A5C8FF')))
            fig.add_trace(go.Scatter(x=x, y=[r['T3_total'] for r in rows], mode='lines+markers', name='T3_total', line=dict(color='#FFB3C6'), marker=dict(color='#FFB3C6')))

        # SMA-smoothed series (solid lines)
        fig.add_trace(go.Scatter(x=x, y=[r.get('sma_T2_scaled') for r in rows], mode='lines', name='SMA T2_scaled', line=dict(color='#FFB3C6')))
        fig.add_trace(go.Scatter(x=x, y=[r.get('sma_T3_total') for r in rows], mode='lines', name='SMA T3', line=dict(color='#FFB3C6')))

        # percent-change traces on secondary axis (dashed lines)
        fig.add_trace(go.Scatter(x=x, y=[r.get('rolling_pctchg_sma_T2_scaled') for r in rows], mode='lines', name='SMA %chg T2 (12)', yaxis='y2', line=dict(color='#FDE68A', dash='dash')))
        fig.add_trace(go.Scatter(x=x, y=[r.get('rolling_pctchg_sma_T3') for r in rows], mode='lines', name='SMA %chg T3 (12)', yaxis='y2', line=dict(color='#FACC15', dash='dash')))

        # Layout matching burnProcessing style
        fig.update_layout(
            title={
                'text': 'OBhr Trends by Tier',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=x_title,
            yaxis=dict(
                title='OBh',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.25)',
                gridwidth=1,
                griddash='dash',
            ),
            yaxis2=dict(
                title='12 Epoch % Change',
                overlaying='y',
                side='right',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.25)',
                gridwidth=1,
                griddash='dot',
            ),
            legend_title='Legend',
            template=None,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            height=600,
            width=1000,
            legend=dict(x=1.05, y=1, xanchor='left'),
        )

        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.25)', gridwidth=1, griddash='dash', zeroline=False)

        # ensure reports dir exists
        out_dir = os.path.dirname(out_html_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # write interactive HTML
        fig.write_html(out_html_path, include_plotlyjs='cdn', full_html=True)
        print(f'Wrote interactive chart HTML to: {out_html_path}')

    # write the HTML chart next to other reports
    reports_html = os.path.join(os.path.dirname(os.path.dirname(out_csv_path)), 'reports', 'obhrs_sma_chart.html')
    # Set show_raw_lines=True to display the raw T2_scaled and T3_total lines with markers
    create_obhrs_chart(rows, out_html_path=reports_html, show_raw_lines=False)

    # write CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'epochId', 'T2_raw', 'T2_scaled', 'T3_total',
            'epoch_all_total', 'epoch_unmapped_total', 'our_sum', 'match',
            'sma_T2_raw', 'sma_T2_scaled', 'sma_T3_total',
            'T2_alltime_pct', 'T3_alltime_pct',
            'rolling_pct_T2', 'rolling_pct_T3',
            'rolling_pctchg_sma_T2_scaled', 'rolling_pctchg_sma_T3'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote totals CSV to: {out_csv_path}')
    # wallets present in the OBhrs metadata that were not filtered into T2 or T3
    print(f'Unique wallets in OBhrs metadata: {len(seen_wallets)}')
    print(f'Wallets not mapped to T2 or T3: {len(unfiltered_wallets)}')
    # report epochs where sums didn't match
    mismatches = [r for r in rows if not r.get('match')]
    if mismatches:
        print('Epoch mismatches (our_sum != epoch_all_total):')
        for m in mismatches:
            print(f" epoch {m['epochId']}: our_sum={m['our_sum']} epoch_total={m['epoch_all_total']} unmapped={m['epoch_unmapped_total']}")
    else:
        print('All epochs matched total obhUsed.')

    # Print unmapped wallet breakdown per epoch
    if unmapped_by_epoch:
        print('\nUnmapped wallets obhUsed per epoch:')
        for eid in sorted(unmapped_by_epoch.keys()):
            print(f' Epoch {eid}:')
            # sort wallets by descending obh for readability
            per = unmapped_by_epoch[eid]
            for w, v in sorted(per.items(), key=lambda x: -x[1]):
                print(f'  {w}: {v}')
    else:
        print('\nNo unmapped wallets recorded.')


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    obhrs_path = os.path.join(base, 'data', 'OBhrs_data', 'OBhrs.json')
    node_summary_csv = os.path.join(base, 'data', 'node_summary.csv')
    out_csv = os.path.join(base, 'data', 'OBhrs_epoch_tier_totals.csv')

    process(obhrs_path, node_summary_csv, out_csv)


if __name__ == '__main__':
    main()

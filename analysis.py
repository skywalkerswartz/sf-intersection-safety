#!/usr/bin/env python3
"""
San Francisco Intersection Safety Analysis
===========================================
Analyzes traffic-normalized crash rates at SF intersections,
broken down by traffic control type (signal, stop sign, none).

Uses a hybrid classification approach that cross-references crash report
control_device fields with signal/stop sign databases to accurately
classify each intersection's traffic control type.

Data Sources:
- Crashes: SF Open Data "Traffic Crashes Resulting in Injury" (ubvf-ztfx)
- Traffic Volumes: SFMTA Intersection Counts 2014-2022
- Traffic Signals: SF Open Data "Traffic Signals" (ybh5-27n2)
- Stop Signs: SF Open Data "Stop Signs" (4542-gpa3)

Usage:
    python3 analysis.py                       # Uses ./data/ directory
    python3 analysis.py --data-dir /path/to   # Custom data directory
    python3 analysis.py --output-dir ./out    # Custom output directory
    python3 analysis.py --json-only           # Only output JSON (for web app)
"""

import argparse
import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

COLORS = {
    'Traffic Signal': '#e74c3c',
    'All-Way Stop': '#3498db',
    '2-Way Stop': '#2ecc71',
    'No Control Device': '#95a5a6',
}

ORDER = ['Traffic Signal', 'All-Way Stop', '2-Way Stop', 'No Control Device']


def normalize_cnn(val):
    """Normalize CNN values to consistent integer strings (remove .0 suffixes etc.)"""
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()


def load_data(data_dir):
    """Load all four data sources from the specified directory."""
    print("=" * 70)
    print("SAN FRANCISCO INTERSECTION SAFETY ANALYSIS")
    print("Traffic-Normalized Crash Rates by Traffic Control Type")
    print("=" * 70)

    print("\n[1] Loading datasets...")

    crashes = pd.read_csv(os.path.join(data_dir, 'crashes.csv'), low_memory=False)
    signals = pd.read_csv(os.path.join(data_dir, 'traffic_signals.csv'), low_memory=False)
    stop_signs = pd.read_csv(os.path.join(data_dir, 'stop_signs.csv'), low_memory=False)
    volumes = pd.read_csv(os.path.join(data_dir, 'intersection_counts.csv'), low_memory=False)

    print(f"  Crashes:             {len(crashes):>8,} records")
    print(f"  Traffic Signals:     {len(signals):>8,} records")
    print(f"  Stop Signs:          {len(stop_signs):>8,} records")
    print(f"  Intersection Counts: {len(volumes):>8,} records")

    return crashes, signals, stop_signs, volumes


def process_volumes(volumes):
    """Compute total approach volume per intersection, aggregated by CNN."""
    print("\n[2] Processing traffic volume data...")

    am_approach_cols = ['AUTO_APPROACH_VOL_E_AM', 'AUTO_APPROACH_VOL_W_AM',
                        'AUTO_APPROACH_VOL_N_AM', 'AUTO_APPROACH_VOL_S_AM']
    pm_approach_cols = ['AUTO_APPROACH_VOL_E_PM', 'AUTO_APPROACH_VOL_W_PM',
                        'AUTO_APPROACH_VOL_N_PM', 'AUTO_APPROACH_VOL_S_PM']

    for col in am_approach_cols + pm_approach_cols:
        volumes[col] = pd.to_numeric(volumes[col], errors='coerce').fillna(0)

    volumes['total_approach_am'] = volumes[am_approach_cols].sum(axis=1)
    volumes['total_approach_pm'] = volumes[pm_approach_cols].sum(axis=1)
    volumes['total_approach'] = volumes['total_approach_am'] + volumes['total_approach_pm']

    # Normalize CNN for joining across datasets
    volumes['CNN'] = volumes['CNN'].apply(normalize_cnn)

    # Average volumes per CNN (some intersections have multiple count records)
    vol_agg = volumes.groupby('CNN').agg(
        avg_total_approach=('total_approach', 'mean'),
        avg_am_approach=('total_approach_am', 'mean'),
        avg_pm_approach=('total_approach_pm', 'mean'),
        num_counts=('total_approach', 'count'),
        primary_st=('PRIMARY_ST', 'first'),
        cross_st=('CROSS_ST', 'first'),
    ).reset_index()

    # Filter out zero-volume and NaN intersections
    vol_agg = vol_agg[vol_agg['avg_total_approach'] > 0]
    vol_agg = vol_agg[vol_agg['primary_st'].notna()]
    vol_agg = vol_agg[vol_agg['CNN'] != 'nan']

    print(f"  Unique intersections with traffic volume data: {len(vol_agg)}")
    print(f"  Volume range: {vol_agg['avg_total_approach'].min():.0f} - "
          f"{vol_agg['avg_total_approach'].max():.0f} vehicles (AM+PM approach)")
    print(f"  Median volume: {vol_agg['avg_total_approach'].median():.0f}")

    return vol_agg


def count_crashes(crashes):
    """Count crashes per intersection with control device info from crash reports."""
    print("\n[3] Counting crashes per intersection...")

    # Focus on crashes at or near intersections
    intersection_crashes = crashes[
        crashes['intersection'].isin([
            'Intersection <= 20ft',
            'Intersection Rear End <= 150ft',
        ])
    ].copy()

    intersection_crashes['cnn_intrsctn_fkey'] = (
        intersection_crashes['cnn_intrsctn_fkey'].apply(normalize_cnn)
    )

    # Aggregate crash counts and extract control device information
    crash_counts = intersection_crashes.groupby('cnn_intrsctn_fkey').agg(
        total_crashes=('unique_id', 'count'),
        total_killed=('number_killed', 'sum'),
        total_injured=('number_injured', 'sum'),
        years_span=('accident_year', lambda x: x.nunique()),
        # Control device info from crash reports (for hybrid classification)
        control_device_mode=('control_device', lambda x: (
            x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        )),
    ).reset_index()
    crash_counts.rename(columns={'cnn_intrsctn_fkey': 'CNN'}, inplace=True)

    # Also compute per-CNN control device distributions for hybrid classification
    ctrl_dist = intersection_crashes.groupby(
        ['cnn_intrsctn_fkey', 'control_device']
    ).size().reset_index(name='count')
    ctrl_dist.rename(columns={'cnn_intrsctn_fkey': 'CNN'}, inplace=True)

    # Calculate fraction of crashes reporting "Functioning" (= traffic signal)
    ctrl_totals = ctrl_dist.groupby('CNN')['count'].sum().reset_index(name='total')
    ctrl_functioning = ctrl_dist[ctrl_dist['control_device'] == 'Functioning']
    ctrl_functioning = ctrl_functioning.groupby('CNN')['count'].sum().reset_index(
        name='functioning_count'
    )
    ctrl_stop = ctrl_dist[ctrl_dist['control_device'] == 'Stop Sign']
    ctrl_stop = ctrl_stop.groupby('CNN')['count'].sum().reset_index(
        name='stop_count'
    )

    ctrl_fractions = ctrl_totals.merge(ctrl_functioning, on='CNN', how='left')
    ctrl_fractions = ctrl_fractions.merge(ctrl_stop, on='CNN', how='left')
    ctrl_fractions['functioning_count'] = ctrl_fractions['functioning_count'].fillna(0)
    ctrl_fractions['stop_count'] = ctrl_fractions['stop_count'].fillna(0)
    ctrl_fractions['frac_functioning'] = (
        ctrl_fractions['functioning_count'] / ctrl_fractions['total']
    )
    ctrl_fractions['frac_stop'] = (
        ctrl_fractions['stop_count'] / ctrl_fractions['total']
    )

    # Add fractions to crash_counts
    crash_counts = crash_counts.merge(
        ctrl_fractions[['CNN', 'frac_functioning', 'frac_stop']],
        on='CNN', how='left',
    )

    # Get lat/lon from crash data
    first_crash = intersection_crashes.groupby('cnn_intrsctn_fkey').agg(
        lat=('tb_latitude', 'first'),
        lon=('tb_longitude', 'first'),
    ).reset_index()
    first_crash.rename(columns={'cnn_intrsctn_fkey': 'CNN'}, inplace=True)
    crash_counts = crash_counts.merge(first_crash, on='CNN', how='left')

    print(f"  Intersection crashes: {len(intersection_crashes):,}")
    print(f"  Unique intersections with crashes: {len(crash_counts):,}")

    return crash_counts


def classify_hybrid(merged, signals, stop_signs):
    """
    Hybrid classification: uses crash report control_device field as primary
    source, supplemented by signal/stop sign databases.

    Logic:
    1. If >50% of crashes at an intersection report "Functioning" -> Traffic Signal
    2. Else if intersection CNN is in the Traffic Signals database -> Traffic Signal
    3. If >30% of crashes report "Stop Sign" -> classify by stop sign count
    4. Else if CNN is in Stop Signs database -> classify by stop sign count
    5. Otherwise -> No Control Device

    This approach is more accurate than using the signal/stop databases alone,
    because the Traffic Signals dataset is significantly incomplete (missing
    ~450 signalized intersections as of 2024).
    """
    print("\n[4] Classifying traffic control types (hybrid approach)...")

    # Build signal CNN set
    signal_types_keep = [
        'SIGNAL', 'CALTRANS', 'CALTRANS - HAWK',
        'CALTRANS (BY CONTRACTOR CONSORTIUM GLC)',
        'SIGNAL (CONTRACTOR MAINTAINED)',
    ]
    active_signals = signals[signals['TYPE'].isin(signal_types_keep)]
    signal_cnns = set(active_signals['CNN'].apply(normalize_cnn))
    print(f"  Signal DB intersections: {len(signal_cnns)}")

    # Build stop sign count per CNN
    stop_signs_copy = stop_signs.copy()
    stop_signs_copy['CNN'] = stop_signs_copy['CNN'].apply(normalize_cnn)
    stops_per_cnn = stop_signs_copy.groupby('CNN').size().reset_index(name='num_stop_signs')
    stop_cnn_map = dict(zip(stops_per_cnn['CNN'], stops_per_cnn['num_stop_signs']))
    stop_cnns = set(stops_per_cnn['CNN'])

    def classify_stop_count(n):
        if n <= 2:
            return '2-Way Stop'
        else:
            return 'All-Way Stop'

    def classify_intersection(row):
        cnn = str(row['CNN']).strip()
        frac_func = row.get('frac_functioning', 0) or 0
        frac_stop = row.get('frac_stop', 0) or 0

        # 1. Crash reports say signal
        if frac_func > 0.5:
            return 'Traffic Signal'

        # 2. Signal database says signal
        if cnn in signal_cnns:
            return 'Traffic Signal'

        # 3. Crash reports say stop sign
        if frac_stop > 0.3:
            n_stops = stop_cnn_map.get(cnn, 2)
            return classify_stop_count(n_stops)

        # 4. Stop sign database
        if cnn in stop_cnns:
            n_stops = stop_cnn_map.get(cnn, 2)
            return classify_stop_count(n_stops)

        # 5. Default
        return 'No Control Device'

    merged['control_type'] = merged.apply(classify_intersection, axis=1)
    merged['control_simple'] = merged['control_type']

    # Report classification distribution
    dist = merged['control_simple'].value_counts()
    print("  Classification results:")
    for ct in ORDER:
        print(f"    {ct}: {dist.get(ct, 0)}")

    return merged


def merge_and_compute(vol_agg, crash_counts, signals, stop_signs):
    """Merge datasets and compute normalized crash rates."""
    print("\n[5] Merging datasets on CNN intersection key...")

    merged = vol_agg.merge(crash_counts, on='CNN', how='left')
    merged['total_crashes'] = merged['total_crashes'].fillna(0)
    merged['total_killed'] = merged['total_killed'].fillna(0)
    merged['total_injured'] = merged['total_injured'].fillna(0)
    merged['frac_functioning'] = merged['frac_functioning'].fillna(0)
    merged['frac_stop'] = merged['frac_stop'].fillna(0)

    # Hybrid classification
    merged = classify_hybrid(merged, signals, stop_signs)

    # Compute normalized crash rate: crashes per 1,000 daily approach vehicles
    merged['crash_rate_per_1k'] = (
        (merged['total_crashes'] / merged['avg_total_approach']) * 1000
    )

    # Annualized rate (roughly 20 years of crash data, 2005-2024)
    data_years = 20
    merged['annual_crash_rate'] = merged['crash_rate_per_1k'] / data_years

    # Injuries per crash
    merged['injuries_per_crash'] = np.where(
        merged['total_crashes'] > 0,
        merged['total_injured'] / merged['total_crashes'],
        0,
    )

    # Only keep intersections with crashes (for meaningful analysis)
    n_with_crashes = (merged['total_crashes'] > 0).sum()
    n_without = (merged['total_crashes'] == 0).sum()
    print(f"  Matched intersections (crashes & volume): {n_with_crashes}")
    print(f"  Intersections with volume but no crashes: {n_without}")
    print(f"  Total merged records: {len(merged)}")

    return merged


def run_statistics(merged):
    """Run statistical tests and print results."""
    print("\n" + "=" * 70)
    print("[6] STATISTICAL ANALYSIS")
    print("=" * 70)

    plot_data = merged[merged['total_crashes'] > 0].copy()

    # Summary by control type
    print("\n--- Crash Rate by Traffic Control Type ---")
    summary = plot_data.groupby('control_simple').agg(
        n_intersections=('CNN', 'count'),
        total_crashes=('total_crashes', 'sum'),
        mean_crashes=('total_crashes', 'mean'),
        median_crashes=('total_crashes', 'median'),
        mean_volume=('avg_total_approach', 'mean'),
        median_volume=('avg_total_approach', 'median'),
        mean_crash_rate=('crash_rate_per_1k', 'mean'),
        median_crash_rate=('crash_rate_per_1k', 'median'),
        mean_annual_rate=('annual_crash_rate', 'mean'),
        median_annual_rate=('annual_crash_rate', 'median'),
        total_killed=('total_killed', 'sum'),
        total_injured=('total_injured', 'sum'),
    ).round(3)
    print(summary.to_string())

    # Statistical tests
    print("\n--- Statistical Tests ---")

    groups = {}
    for ct in merged['control_simple'].unique():
        data = merged[merged['control_simple'] == ct]['crash_rate_per_1k'].dropna()
        if len(data) >= 5:
            groups[ct] = data

    if len(groups) >= 2:
        # Kruskal-Wallis test (non-parametric ANOVA)
        group_arrays = list(groups.values())
        stat, p = stats.kruskal(*group_arrays)
        print(f"\nKruskal-Wallis H-test across all control types:")
        print(f"  H-statistic: {stat:.4f}, p-value: {p:.6f}")
        if p < 0.05:
            print("  -> Significant difference between groups (p < 0.05)")
        else:
            print("  -> No significant difference between groups (p >= 0.05)")

        # Pairwise Mann-Whitney U tests
        print("\nPairwise Mann-Whitney U tests:")
        group_names = list(groups.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                stat, p = stats.mannwhitneyu(
                    groups[g1], groups[g2], alternative='two-sided'
                )
                sig = (
                    "***" if p < 0.001
                    else "**" if p < 0.01
                    else "*" if p < 0.05
                    else "ns"
                )
                print(f"  {g1} vs {g2}: U={stat:.0f}, p={p:.6f} {sig}")

    # Correlation between volume and crash count
    corr_pearson, p_pearson = stats.pearsonr(
        merged['avg_total_approach'], merged['total_crashes']
    )
    corr_spearman, p_spearman = stats.spearmanr(
        merged['avg_total_approach'], merged['total_crashes']
    )
    print(f"\nCorrelation: Traffic Volume vs Total Crashes")
    print(f"  Pearson r:  {corr_pearson:.4f}  (p={p_pearson:.6f})")
    print(f"  Spearman rho: {corr_spearman:.4f}  (p={p_spearman:.6f})")

    # Severity analysis
    print("\n--- Severity Analysis by Control Type ---")
    severity = plot_data.groupby('control_simple').agg(
        mean_injuries_per_crash=('injuries_per_crash', 'mean'),
        total_fatalities=('total_killed', 'sum'),
        total_crashes=('total_crashes', 'sum'),
    ).round(3)
    severity['fatalities_per_1000_crashes'] = (
        (severity['total_fatalities'] / severity['total_crashes'] * 1000).round(2)
    )
    print(severity.to_string())

    # Top 20 most dangerous intersections
    print("\n--- Top 20 Intersections by Traffic-Normalized Crash Rate ---")
    top20 = merged.nlargest(20, 'crash_rate_per_1k')[
        ['primary_st', 'cross_st', 'avg_total_approach', 'total_crashes',
         'control_simple', 'crash_rate_per_1k', 'annual_crash_rate']
    ]
    top20.columns = [
        'Primary St', 'Cross St', 'Avg Daily Volume', 'Total Crashes',
        'Control Type', 'Crashes/1K Veh (Total)', 'Crashes/1K Veh (Annual)',
    ]
    print(top20.to_string(index=False))

    return summary, corr_pearson


def export_json(merged, output_dir):
    """Export intersection data as JSON for the web app."""
    records = []
    for _, row in merged.iterrows():
        if row['total_crashes'] > 0:
            records.append({
                'cnn': row['CNN'],
                'primary_st': row['primary_st'],
                'cross_st': row['cross_st'],
                'daily_volume': int(round(row['avg_total_approach'])),
                'total_crashes': int(row['total_crashes']),
                'fatalities': int(row['total_killed']),
                'injuries': int(row['total_injured']),
                'control_type': row['control_type'],
                'control_simple': row['control_simple'],
                'crash_rate_per_1k': round(row['crash_rate_per_1k'], 2),
                'annual_crash_rate': round(row['annual_crash_rate'], 2),
                'injuries_per_crash': round(row['injuries_per_crash'], 2),
                'lat': row.get('lat', None),
                'lon': row.get('lon', None),
            })

    json_path = os.path.join(output_dir, 'intersection_data.json')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"  Saved: {json_path} ({len(records)} intersections)")
    return records


def export_csv(merged, summary, output_dir):
    """Export CSV files."""
    export_cols = [
        'CNN', 'primary_st', 'cross_st', 'avg_total_approach', 'total_crashes',
        'total_killed', 'total_injured', 'control_type', 'control_simple',
        'crash_rate_per_1k', 'annual_crash_rate',
    ]
    csv_path = os.path.join(output_dir, 'intersection_crash_rates.csv')
    merged[export_cols].sort_values(
        'crash_rate_per_1k', ascending=False
    ).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary_path = os.path.join(output_dir, 'summary_by_control_type.csv')
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path}")


def create_visualizations(merged, corr_pearson, output_dir):
    """Generate static PNG charts."""
    print("\n[8] Creating visualizations...")

    plot_data = merged[merged['total_crashes'] > 0].copy()
    order = [o for o in ORDER if o in plot_data['control_simple'].unique()]

    # --- Figure 1: Box + Violin plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.boxplot(x='control_simple', y='crash_rate_per_1k',
                data=plot_data, order=order, ax=axes[0],
                palette=COLORS, showfliers=False, width=0.6)
    axes[0].set_xlabel('Traffic Control Type', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Crashes per 1,000 Daily Vehicles\n(2005-2024 Total)', fontsize=12)
    axes[0].set_title('Crash Rate Distribution by Control Type',
                      fontsize=14, fontweight='bold')
    for i, ct in enumerate(order):
        n = len(plot_data[plot_data['control_simple'] == ct])
        axes[0].text(i, axes[0].get_ylim()[1] * 0.95, f'n={n}',
                     ha='center', fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)

    sns.violinplot(x='control_simple', y='crash_rate_per_1k',
                   data=plot_data, order=order, ax=axes[1],
                   palette=COLORS, inner=None, alpha=0.3, cut=0)
    sns.stripplot(x='control_simple', y='crash_rate_per_1k',
                  data=plot_data, order=order, ax=axes[1],
                  palette=COLORS, alpha=0.4, size=4, jitter=True)
    for i, ct in enumerate(order):
        med = plot_data[plot_data['control_simple'] == ct]['crash_rate_per_1k'].median()
        axes[1].plot(i, med, 'k_', markersize=20, markeredgewidth=3)
    axes[1].set_xlabel('Traffic Control Type', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Crashes per 1,000 Daily Vehicles\n(2005-2024 Total)', fontsize=12)
    axes[1].set_title('Crash Rate Distribution (Violin + Strip)',
                      fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_crash_rate_by_control_type.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_crash_rate_by_control_type.png")

    # --- Figure 2: Scatter plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for ct in order:
        subset = plot_data[plot_data['control_simple'] == ct]
        ax.scatter(subset['avg_total_approach'], subset['total_crashes'],
                   alpha=0.5, s=30, label=ct, color=COLORS.get(ct, '#333'),
                   edgecolors='white', linewidth=0.3)
    ax.set_xlabel('Average Daily Approach Volume (Vehicles)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Crashes (2005-2024)', fontsize=13, fontweight='bold')
    ax.set_title('Traffic Volume vs. Crash Count by Control Type',
                 fontsize=15, fontweight='bold')

    z = np.polyfit(plot_data['avg_total_approach'], plot_data['total_crashes'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(plot_data['avg_total_approach'].min(),
                          plot_data['avg_total_approach'].max(), 100)
    ax.plot(x_range, p(x_range), '--', color='black', alpha=0.7, linewidth=2,
            label=f'Trend (r={corr_pearson:.3f})')
    ax.legend(title='Control Type', fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_volume_vs_crashes.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_volume_vs_crashes.png")

    # --- Figure 3: Bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    rate_summary = plot_data.groupby('control_simple').agg(
        mean_rate=('crash_rate_per_1k', 'mean'),
        median_rate=('crash_rate_per_1k', 'median'),
        n=('CNN', 'count'),
    ).reindex(order)

    bars1 = axes[0].bar(range(len(order)), rate_summary['mean_rate'],
                        color=[COLORS[o] for o in order],
                        edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=20, ha='right')
    axes[0].set_ylabel('Mean Crashes per 1,000 Daily Vehicles', fontsize=12)
    axes[0].set_title('Mean Crash Rate by Control Type',
                      fontsize=14, fontweight='bold')
    for bar, val in zip(bars1, rate_summary['mean_rate']):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    bars2 = axes[1].bar(range(len(order)), rate_summary['median_rate'],
                        color=[COLORS[o] for o in order],
                        edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=20, ha='right')
    axes[1].set_ylabel('Median Crashes per 1,000 Daily Vehicles', fontsize=12)
    axes[1].set_title('Median Crash Rate by Control Type',
                      fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, rate_summary['median_rate']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_mean_median_rates.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_mean_median_rates.png")

    # --- Figure 4: Severity ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sev_data = plot_data.copy()
    sev_data['injury_rate'] = sev_data['total_injured'] / sev_data['total_crashes']
    sev_summary = sev_data.groupby('control_simple').agg(
        mean_injury_rate=('injury_rate', 'mean'),
        total_fatalities=('total_killed', 'sum'),
        total_crashes=('total_crashes', 'sum'),
    ).reindex(order)

    bars3 = axes[0].bar(range(len(order)), sev_summary['mean_injury_rate'],
                        color=[COLORS[o] for o in order],
                        edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=20, ha='right')
    axes[0].set_ylabel('Mean Injuries per Crash', fontsize=12)
    axes[0].set_title('Crash Severity: Injuries per Crash',
                      fontsize=14, fontweight='bold')
    for bar, val in zip(bars3, sev_summary['mean_injury_rate']):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    sev_summary['fat_per_1k'] = (
        sev_summary['total_fatalities'] / sev_summary['total_crashes'] * 1000
    )
    bars4 = axes[1].bar(range(len(order)), sev_summary['fat_per_1k'],
                        color=[COLORS[o] for o in order],
                        edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=20, ha='right')
    axes[1].set_ylabel('Fatalities per 1,000 Crashes', fontsize=12)
    axes[1].set_title('Crash Severity: Fatality Rate',
                      fontsize=14, fontweight='bold')
    for bar, val in zip(bars4, sev_summary['fat_per_1k']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_severity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_severity.png")


def main():
    parser = argparse.ArgumentParser(
        description='SF Intersection Safety Analysis'
    )
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing CSV data files (default: data/)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory for output files (default: output/)')
    parser.add_argument('--json-only', action='store_true',
                        help='Only export JSON (skip charts and statistics)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    crashes, signals, stop_signs, volumes = load_data(args.data_dir)

    # Process
    vol_agg = process_volumes(volumes)
    crash_counts = count_crashes(crashes)
    merged = merge_and_compute(vol_agg, crash_counts, signals, stop_signs)

    # Export JSON (always)
    print("\n[7] Exporting data...")
    records = export_json(merged, args.output_dir)

    if not args.json_only:
        # Statistics
        summary, corr_pearson = run_statistics(merged)

        # CSV exports
        export_csv(merged, summary, args.output_dir)

        # Visualizations
        create_visualizations(merged, corr_pearson, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Intersections analyzed: {len(records)}")
    print(f"  Output directory: {args.output_dir}/")


if __name__ == '__main__':
    main()

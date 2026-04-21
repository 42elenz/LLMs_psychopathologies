import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

df = pd.read_csv('/zi/home/esra.lenz/Documents/00_HITKIP/01_GPTS/99_Git_share/LLMs_psychopatholgies/outputs/tables/all_clinicians_ratings.csv')
ref_df = pd.read_csv('/zi/home/esra.lenz/Documents/00_HITKIP/01_GPTS/99_Git_share/LLMs_psychopatholgies/data/00_ratings/reference.csv')
ref_df = ref_df.set_index('ID_Video')

# Extract AMDP item columns
item_cols = [c for c in df.columns if c.startswith('p') and c[1].isdigit()]
item_nums = [int(c.split('_')[0][1:]) for c in item_cols]

outdir = '/zi/home/esra.lenz/Documents/00_HITKIP/01_GPTS/99_Git_share/LLMs_psychopatholgies/outputs/clinician_rating_sheets'
os.makedirs(outdir, exist_ok=True)

# Define AMDP domains with approximate boundaries
domains = [
    (1, 4, 'Consciousness', '#e6f2ff'),
    (5, 8, 'Orientation', '#d6eaff'),
    (9, 14, 'Attention\n& Memory', '#fff2e6'),
    (15, 26, 'Formal\nThought', '#e6ffe6'),
    (27, 32, 'Fears &\nCompulsions', '#ffe6e6'),
    (33, 46, 'Delusions', '#ffd6d6'),
    (47, 52, 'Perceptual\nDisturbances', '#ffcccc'),
    (53, 58, 'Ego\nDisturbances', '#f5e6ff'),
    (59, 79, 'Affectivity', '#f2e6ff'),
    (80, 88, 'Drive &\nPsychomotility', '#e6ffff'),
    (89, 91, 'Circadian', '#d6ffff'),
    (92, 100, 'Other', '#e6fff2'),
]

### Compute per-rater quality metrics and flag anomalies
quality_records = []

for _, row in df.iterrows():
    rater = row['rater_id']
    video = row['video_name']
    acc = row['accuracy']
    
    raw = np.array([row[c] for c in item_cols], dtype=float)
    na_mask = raw == -99
    
    # Map: 0-3 stay, -99 -> -0.5 (displayed as "NA")
    ratings = raw.copy()
    ratings[na_mask] = -0.5
    
    # Reference ratings for this video
    vid = row['video_id']
    ref_raw = np.array([ref_df.loc[vid, c] for c in item_cols], dtype=float)
    ref_na_mask = ref_raw == -99
    ref_ratings = ref_raw.copy()
    ref_ratings[ref_na_mask] = -0.5
    
    # Recompute accuracy: exact match on full 0-3 scale, NAs (-99) count as a value
    acc_full = np.sum(raw == ref_raw) / len(raw)

    # --- Engagement / quality metrics ---
    answered = raw[~na_mask]       # only non-NA responses
    n_answered = len(answered)
    pct_na = na_mask.sum() / len(raw) * 100

    if n_answered > 0:
        # Straight-lining: how often did rater pick the single most frequent value?
        values, counts = np.unique(answered, return_counts=True)
        most_common_val = values[counts.argmax()]
        most_common_pct = counts.max() / n_answered * 100
        n_unique = len(values)
        rating_std = np.std(answered)
        mean_rating = np.mean(answered)
        pct_zero = np.mean(answered == 0) * 100

        # Longest run of identical consecutive ratings
        longest_run = 1
        current_run = 1
        for i in range(1, n_answered):
            if answered[i] == answered[i - 1]:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
    else:
        most_common_val = most_common_pct = n_unique = np.nan
        rating_std = mean_rating = pct_zero = np.nan
        longest_run = 0

    # Comparison with reference (on jointly valid items)
    valid = ~na_mask & ~ref_na_mask
    n_valid = valid.sum()
    if n_valid > 0:
        diff = raw[valid] - ref_raw[valid]
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
    else:
        mae = bias = np.nan

    # --- Flags for disengagement / careless responding ---
    flags = []
    if most_common_pct and most_common_pct > 80:
        flags.append(f'straight-line ({int(most_common_val)}={most_common_pct:.0f}%)')
    if n_unique == 1 and n_answered > 10:
        flags.append('single_value_only')
    if rating_std < 0.3 and n_answered > 10:
        flags.append('near_zero_variance')
    if longest_run > 20:
        flags.append(f'long_run ({longest_run})')
    if pct_na > 50:
        flags.append(f'excessive_NA ({pct_na:.0f}%)')
    if pct_zero > 85:
        flags.append(f'mostly_zeros ({pct_zero:.0f}%)')
    if mae > 1.5:
        flags.append(f'high_MAE ({mae:.2f})')
    
    quality_records.append({
        'rater_id': rater, 'video_id': vid, 'video_name': video,
        'accuracy': round(acc_full, 3), 'mae': round(mae, 3) if not np.isnan(mae) else np.nan,
        'bias': round(bias, 3) if not np.isnan(bias) else np.nan,
        'rating_std': round(rating_std, 3) if not np.isnan(rating_std) else np.nan,
        'mean_rating': round(mean_rating, 2) if not np.isnan(mean_rating) else np.nan,
        'most_common_value': int(most_common_val) if not np.isnan(most_common_val) else np.nan,
        'most_common_pct': round(most_common_pct, 1) if not np.isnan(most_common_pct) else np.nan,
        'n_unique_values': int(n_unique) if not np.isnan(n_unique) else np.nan,
        'longest_run': longest_run,
        'pct_na': round(pct_na, 1), 'pct_zero': round(pct_zero, 1) if not np.isnan(pct_zero) else np.nan,
        'n_flags': len(flags),
        'flags': '; '.join(flags) if flags else ''
    })
    
    fig, ax = plt.subplots(figsize=(20, 5))
    
    # Domain background shading (alternating grey/white)
    for i, (start, end, label, _) in enumerate(domains):
        idx_start = item_nums.index(start) if start in item_nums else 0
        idx_end = item_nums.index(end) if end in item_nums else len(item_nums)-1
        bg = '#f0f0f0' if i % 2 == 0 else 'white'
        ax.axvspan(idx_start - 0.5, idx_end + 0.5, color=bg, zorder=0)
        mid = (idx_start + idx_end) / 2
        ax.text(mid, 3.35, label, ha='center', va='bottom', fontsize=6, fontweight='bold', color='#777')
    
    # Connected lines + scatter
    x = np.arange(len(ratings))
    ax.plot(x, ratings, '-', color='black', alpha=0.4, linewidth=0.8, zorder=1)
    ax.scatter(x, ratings, s=18, color='black', edgecolor='white', linewidth=0.3, zorder=2)
    ax.plot(x, ref_ratings, '-', color='#ff7f0e', alpha=0.4, linewidth=0.8, zorder=1)
    ax.scatter(x, ref_ratings, s=18, color='#ff7f0e', edgecolor='white', linewidth=0.3, zorder=2, marker='^')
    
    ax.set_ylim(-0.9, 3.8)
    ax.set_yticks([-0.5, 0, 1, 2, 3])
    ax.set_yticklabels(['NA', '0\nnot present', '1\nmild', '2\nmoderate', '3\nsevere'], fontsize=8)
    ax.set_xlim(-1, len(ratings))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in item_nums], fontsize=5, rotation=90)
    ax.set_xlabel('AMDP Item', fontsize=10)
    ax.set_ylabel('Rating', fontsize=10)
    ax.set_title(f'Clinician: {rater}  |  Video: {video}  |  Accuracy: {acc_full:.0%}  |  NA: {int(row["n_na"])}', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5, alpha=0.3)
    
    # Small legend
    handles = [
        plt.Line2D([0],[0], marker='o', color='black', markersize=5, linestyle='-', alpha=0.6, label='Clinician'),
        plt.Line2D([0],[0], marker='^', color='#ff7f0e', markersize=5, linestyle='-', alpha=0.6, label='Reference'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.5)
    
    plt.tight_layout()
    safe_name = rater.replace('/', '_').replace(' ', '_')
    fig.savefig(f'{outdir}/{safe_name}_{video}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"Done. {len(df)} sheets saved to {outdir}")

# --- Save quality metrics & print flagged raters ---
qdf = pd.DataFrame(quality_records)
qdf = qdf.sort_values('n_flags', ascending=False)
quality_path = os.path.join(os.path.dirname(outdir), 'tables', 'rater_engagement_flags.csv')
os.makedirs(os.path.dirname(quality_path), exist_ok=True)
qdf.to_csv(quality_path, index=False)
print(f"\nEngagement metrics saved to {quality_path}")

flagged = qdf[qdf['n_flags'] > 0]
if len(flagged) > 0:
    print(f"\n⚠ {len(flagged)} rater-video pairs flagged for potential disengagement:\n")
    for _, r in flagged.iterrows():
        print(f"  {r['rater_id']:40s}  {r['video_name']:15s}  acc={r['accuracy']:.0%}  "
              f"std={r['rating_std']:.2f}  most_common={r['most_common_pct']:.0f}%  "
              f"run={r['longest_run']}  → {r['flags']}")
else:
    print("\nNo raters flagged for disengagement.")
import os
import pandas as pd
import re
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import math
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, binomtest
from itertools import combinations

AMDP_MAPPING_English = {
    1:  ("bewusstseinsverminderung", "Lowered_Vigilance"),
    2:  ("bewusstseinstruebung", "Clouded_Consciousness"),
    3:  ("bewusstseinseinengung", "Narrowed_Consciousness"),
    4:  ("bewusstseinsverschiebung", "Expanded_Consciousness"),

    5:  ("zeitliche orientierungsstoerung", "Disorientation_for_Time"),
    6:  ("oertliche orientierungsstoerung", "Disorientation_for_Place"),
    7:  ("situative orientierungsstoerung", "Disorientation_for_Situation"),
    8:  ("orientierungsstoerung eigene person", "Disorientation_for_Person"),

    9:  ("auffassungsstoerungen", "Disturbed_Apperception"),
    10: ("konzentrationsstoerungen", "Disturbed_Concentration"),
    11: ("merkfaehigkeitsstoerungen", "Disturbed_Short-Term_Memory"),
    12: ("gedaechtnisstoerungen", "Disturbed_Long-Term_Memory"),
    13: ("konfabulationen", "Confabulation"),
    14: ("paramnesien", "Paramnesias"),

    15: ("gehemmt", "Inhibited_Thinking"),
    16: ("verlangsamt", "Retarded_Thinking"),
    17: ("umstaendlich", "Circumstantial_Thinking"),
    18: ("eingeengt", "Restricted_Thinking"),
    19: ("perseverierend", "Perseverative_Thinking"),
    20: ("grueblen", "Rumination"),
    21: ("gedankendraengen", "Pressured_Thinking"),
    22: ("ideenfluechtig", "Flight_of_Ideas"),
    23: ("vorbeireden", "Tangential_Thinking"),
    24: ("gesperrt", "Thought_Blocking"),
    25: ("inkohaerent", "Incoherence_Derailment"),
    26: ("neologismen", "Neologisms"),

    27: ("misstrauen", "Suspiciousness"),
    28: ("hypochondrie", "Hypochondriasis"),
    29: ("phobien", "Phobias"),
    30: ("zwangsdenken", "Obsessive_Thoughts"),
    31: ("zwangsimpulse", "Compulsive_Impulses"),
    32: ("zwangshandlungen", "Compulsive_Actions"),

    33: ("wahnstimmung", "Delusional_Mood"),
    34: ("wahnwahrnehmung", "Delusional_Perception"),
    35: ("wahneinfall", "Sudden_Delusional_Ideas"),
    36: ("wahngedanken", "Delusional_Ideas"),
    37: ("systematisierter wahn", "Systematized_Delusions"),
    38: ("wahndynamik", "Delusional_Dynamics"),
    39: ("beziehungswahn", "Delusions_of_Reference"),
    40: ("verfolgungswahn", "Delusions_of_Persecution"),
    41: ("eifersuchtswahn", "Delusions_of_Jealousy"),
    42: ("schuldwahn", "Delusions_of_Guilt"),
    43: ("verarmungswahn", "Delusions_of_Impoverishment"),
    44: ("hypochondrischer wahn", "Hypochondriacal_Delusions"),
    45: ("groessenwahn", "Delusions_of_Grandiosity"),
    46: ("andere wahn", "Other_Delusions"),

    47: ("illusionen", "Illusions"),
    48: ("stimmenhoeren", "Hearing_Voices"),
    49: ("akustische halluzinationen", "Other_Auditory_Hallucinations"),
    50: ("optische halluzinationen", "Visual_Hallucinations"),
    51: ("koerperhalluzinationen", "Bodily_Hallucinations"),
    52: ("geruchs", "Olfactory_and_Gustatory_Hallucinations"),
    53: ("derealisation", "Derealization"),
    54: ("depersonalisation", "Depersonalization"),
    55: ("gedankenausbreitung", "Thought_Broadcasting"),
    56: ("gedankenentzug", "Thought_Withdrawal"),
    57: ("gedankeneingebung", "Thought_Insertion"),
    58: ("fremdbeeinflussung", "Other_Feelings_of_Alien_Influence"),

    59: ("ratlos", "Perplexity"),
    60: ("gefuehllosigkeit", "Feeling_of_Loss_of_Feeling"),
    61: ("affektarm", "Blunted_Affect"),
    62: ("vitalgefuehle", "Felt_Loss_of_Vitality"),
    63: ("deprimiert", "Depressed_Mood"),
    64: ("hoffnungslos", "Hopelessness"),
    65: ("aengstlich", "Anxiety"),
    66: ("euphorisch", "Euphoria"),
    67: ("dysphorisch", "Dysphoria"),
    68: ("gereizt", "Irritability"),
    69: ("unruhig", "Inner_Restlessness"),
    70: ("klagsam", "Complaintiveness"),

    71: ("insuffizienzgefuehle", "Feelings_of_Inadequacy"),
    72: ("selbstwertgefuehl", "Exaggerated_Self-Esteem"),
    73: ("schuldgefuehle", "Feelings_of_Guilt"),
    74: ("verarmungsgefuehle", "Feelings_of_Impoverishment"),
    75: ("ambivalent", "Ambivalence"),
    76: ("parathymie", "Parathymia"),
    77: ("affektlabil", "Affective_Lability"),
    78: ("affektinkontinent", "Affective_Incontinence"),
    79: ("affektstarr", "Affective_Rigidity"),
    80: ("antriebsarm", "Lack_of_Drive"),
    81: ("antriebsgehemmt", "Inhibition_of_Drive"),
    82: ("antriebsgesteigert", "Increased_Drive"),
    83: ("motorisch unruhig", "Motor_Restlessness"),
    84: ("parakines", "Parakinesia"),
    85: ("maniriert", "Mannerisms"),
    86: ("theatralisch", "Histrionics"),
    87: ("mutistisch", "Mutism"),
    88: ("logorrhoe", "Logorrhoea"),

    89: ("morgens schlechter", "Worse_in_the_Morning"),
    90: ("abends schlechter", "Worse_in_the_Evening"),
    91: ("abends besser", "Better_in_the_Evening"),

    92: ("sozialer rueckzug", "Social_Withdrawal"),
    93: ("soziale umtriebigkeit", "Excessive_Social_Contact"),
    94: ("aggressiv", "Aggressiveness"),
    95: ("suizidal", "Suicidal_Behaviour"),
    96: ("selbstbeschaedigung", "Self-Harm"),
    97: ("krankheitsgefuehl", "Lack_of_Feeling_Ill"),
    98: ("krankheitseinsicht", "Lack_of_Insight_Into_Illness"),
    99: ("ablehnung behandlung", "Uncooperativeness"),
    100: ("pflegebeduerftig", "Need_for_Care"),
}
def map_ppb_to_categories(row):
    if row == 0:
        return 0
    elif row in [1, 2, 3]:
        return 1
    elif row == -99:
        return -99
    else:
        return np.nan

def map_ki_rating(row):
    if row == -1:
        return -99

def reduce_data_to_3_categories(psy_cols, dfs, col_suffix="_reduced"):
    dfs_result = []
    for df in dfs:
        df_copy = df.copy()
        new_reduced = df_copy[psy_cols].applymap(map_ppb_to_categories)
        new_reduced.columns = [c + col_suffix for c in new_reduced.columns]
        df_copy = pd.concat([df_copy, new_reduced], axis=1)
        df_copy = df_copy.copy()
        dfs_result.append(df_copy)
    return dfs_result

def map_psy_columns_to_english(
    df: pd.DataFrame,
    psy_cols_regex: str = r'^p(\d+)_(.+?)(_final_reduced|_final|_reduced|)?$',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Map psychology column names to English AMDP terms.
    
    Parameters:
    -----------
    df : DataFrame
        Your dataframe with columns like 'p1_bewusstseinsverminderung_final'
    amdp_mapping : dict
        Dictionary mapping {item_number: (german_name, english_name)}
    psy_cols_regex : str
        Regex pattern to extract item numbers from column names
    inplace : bool
        If True, modifies df in place. If False, returns new df.
    
    Returns:
    --------
    DataFrame with renamed columns
    """
    import re
    
    if not inplace:
        df = df.copy()
    
    # Build mapping from old column names to new names
    column_mapping = {}
    
    for col in df.columns:
        if "begründung" in col:
            continue
        # Match pattern: p{num}_{german_name}_{suffix}
        match = re.match(psy_cols_regex, col)
        if match:
            item_num = int(match.group(1))
            german_part = match.group(2)  # The German name part
            suffix = match.group(3) if match.group(3) else ''  # e.g., '_final', '_final_reduced'
            
            # Get english name from AMDP_MAPPING
            if item_num in AMDP_MAPPING_English:
                _, english_name = AMDP_MAPPING_English[item_num]
                
                # Build new column name with English term
                new_col = f'p{item_num}_{english_name.lower().replace(" ", "_").replace("/", "_")}{suffix}'
                column_mapping[col] = new_col
    
    # Rename columns
    df.rename(columns=column_mapping, inplace=True)
    
    return df


def extract_psy_cols(df, psy_cols_regex=None):
    if psy_cols_regex is None:
        psy_cols_regex = r'^p\d+_.*_final$'
    return [c for c in df.columns if re.match(psy_cols_regex, c)]

def human_vs_reference_df(
    human_master: pd.DataFrame,
    reference: pd.DataFrame,
    psy_cols: list,
    video_ids: list = None,
    human_video_col: str = "video_id",
    ref_video_col: str = "ID_Video",
    rating_levels: list = None
) -> pd.DataFrame:
    """
    Return a DataFrame comparing human ratings to the reference per video and item.
    
    Columns:
      - video_id
      - item
      - reference
      - human_total
      - human_mean
      - human_correct_n
      - human_correct_pct
      - count_0, count_1, count_2, count_3, count_-99 (how often each rating was given)
    """
    if rating_levels is None:
        rating_levels = [0, 1, 2, 3, -99]
    
    if video_ids is None:
        video_ids = sorted(
            set(human_master[human_video_col].dropna().unique())
            & set(reference[ref_video_col].dropna().unique())
        )

    results = []

    for vid in video_ids:
        human_vid = human_master[human_master[human_video_col] == vid]
        ref_vid = reference[reference[ref_video_col] == vid]

        if human_vid.empty or ref_vid.empty:
            continue

        for item in psy_cols:
            if item not in human_vid.columns or item not in ref_vid.columns:
                continue

            # Reference rating (take first row)
            ref_series = ref_vid[item]
            ref_rating = ref_series.iloc[0] if len(ref_series) > 0 else np.nan

            # Human ratings (numeric)
            hr = pd.to_numeric(human_vid[item], errors="coerce").dropna()
            human_total = int(len(hr))
            human_mean = float(hr.mean()) if human_total > 0 else np.nan

            # Agreement with reference
            if pd.notna(ref_rating) and human_total > 0:
                human_correct_n = int((hr == ref_rating).sum())
                human_correct_pct = (human_correct_n / human_total) * 100
            else:
                human_correct_n = np.nan
                human_correct_pct = np.nan

            # Count how often each rating level was given
            rating_counts = {}
            for level in rating_levels:
                count = int((hr == level).sum())
                rating_counts[f"count_{level}"] = count

            # Build result row
            result_row = {
                "video_id": vid,
                "item": item,
                "reference": ref_rating,
                "human_total": human_total,
                "human_mean": human_mean,
                "human_correct_n": human_correct_n,
                "human_correct_pct": human_correct_pct
            }
            
            # Add rating counts
            result_row.update(rating_counts)
            
            results.append(result_row)

    return pd.DataFrame(results).sort_values(["video_id", "item"]).reset_index(drop=True)



def plot_multi_site_accuracies_split(
    groups_dict: dict,
    site_accuracies: dict,
    site_mode_accuracies: dict,
    site_config: dict,
    index: int = 1,
    title: str = None,
    show_error_bars: bool = True,
    show_std_values: bool = False,
    show_annotations: bool = True,
    annotate_best_only: bool = True,
    label_y: str = "Prozentuale Übereinstimmung mit Referenz",
    fontsize_x: int = 12,
    fontsize_y: int = 10,
    font_x_orientation: str = "right",
    fontsize_annotations: int = 8,
    fontweight_annotation: str = "bold",
    hide_ai_col: bool = False,
    title_fontsize: int = 16,
    max_sites_per_plot: int = 4,  # <--- NEW PARAMETER
):
    """
    Plot accuracies for multiple sites with individual and majority vote results.
    Automatically splits into multiple subplots if there are too many sites.
    """
    labels = list(groups_dict.keys())
    n_groups = len(labels)

    # Extract data for all sites
    site_data = {}
    for site_key in site_accuracies.keys():
        site_individual = [float(site_accuracies[site_key][g][index]) for g in labels]
        site_majority = [float(site_mode_accuracies[site_key][g][index]) for g in labels]
        site_std = [site_accuracies[site_key][g][0].std() for g in labels]
        
        if site_key not in site_config:
            continue

        site_data[site_key] = {
            "individual": site_individual,
            "majority": site_majority,
            "std": site_std,
            "name": site_config[site_key]["name"],
            "color_individual": site_config[site_key]["color_individual"],
            "color_majority": site_config[site_key]["color_majority"],
            "is_ai": site_key == "ki",
        }

    # Calculate number of subplots needed
    n_sites = len(site_data)
    n_subplots = math.ceil(n_sites / max_sites_per_plot)
    
    # Split sites into chunks
    site_items = list(site_data.items())
    site_chunks = [site_items[i:i + max_sites_per_plot] 
                   for i in range(0, len(site_items), max_sites_per_plot)]
    
    # Create figure with subplots
    fig_height = 8 * n_subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, fig_height))
    
    # Ensure axes is always iterable
    if n_subplots == 1:
        axes = [axes]
    
    # Process each subplot
    all_figs_axes = []
    
    for subplot_idx, (ax, site_chunk) in enumerate(zip(axes, site_chunks)):
        chunk_dict = dict(site_chunk)
        n_sites_chunk = len(chunk_dict)
        
        # --- dynamic y-axis limits ---
        all_values = []
        for data in chunk_dict.values():
            all_values.extend(data["individual"])
            all_values.extend(data["majority"])

        min_val = min(all_values) * 0.95
        max_val = max(all_values) * 1.05
        y_range = max_val - min_val

        if y_range < 0.1:
            mid_val = (min_val + max_val) / 2
            min_val = mid_val - 0.05
            max_val = mid_val + 0.05

        # --- calculate bar positions ---
        total_bars = n_sites_chunk * 2
        spacing = 1.5
        x = np.arange(n_groups) * spacing
        group_spacing = 0.05
        available_width = spacing * 0.8
        bar_width = available_width / total_bars

        # --- precompute best value per group ---
        best_per_group = None
        if show_annotations and annotate_best_only:
            best_per_group = []
            for j in range(n_groups):
                vals_j = []
                for data in chunk_dict.values():
                    vals_j.append(data["individual"][j])
                    vals_j.append(data["majority"][j])
                best_per_group.append(max(vals_j))

        # --- plotting ---
        current_offset = -(available_width / 2)

        for i, (site_key, data) in enumerate(chunk_dict.items()):
            pos_individual = x + current_offset
            pos_majority = x + current_offset + bar_width

            if hide_ai_col:
                site_name_lower = data["name"].lower()
                ai_keywords = ["openai", "google", "llama", "gemini", "gpt", "claude"]
                if any(keyword in site_name_lower for keyword in ai_keywords):
                    current_offset += 2 * bar_width + group_spacing
                    continue

            # individual bars
            bars_ind = ax.bar(
                pos_individual, data["individual"], bar_width,
                yerr=data["std"] if show_error_bars else None,
                label=f"{data['name']} - Einzelbewertung",
                color=data["color_individual"], alpha=0.8,
                edgecolor="white", linewidth=1,
                capsize=5 if show_error_bars else 0,
                error_kw={"elinewidth": 1, "capthick": 1} if show_error_bars else {},
            )

            # majority bars
            bars_maj = ax.bar(
                pos_majority, data["majority"], bar_width,
                label=f"{data['name']} - Mehrheitsvotum",
                color=data["color_majority"], alpha=0.8,
                edgecolor="white", linewidth=1,
            )

            # --- annotations ---
            if show_annotations:
                tol = 1e-6
                for j, (bar_ind, bar_maj) in enumerate(zip(bars_ind, bars_maj)):
                    height_ind = bar_ind.get_height()
                    height_maj = bar_maj.get_height()

                    annotate_ind = True
                    annotate_maj = True
                    if annotate_best_only and best_per_group is not None:
                        annotate_ind = abs(height_ind - best_per_group[j]) < tol
                        annotate_maj = abs(height_maj - best_per_group[j]) < tol

                    if annotate_ind:
                        text_ind = f'{height_ind*100:.1f}%\n±{data["std"][j]*100:.1f}' if show_std_values else f'{height_ind*100:.1f}%'
                        ax.annotate(
                            text_ind,
                            xy=(bar_ind.get_x() + bar_ind.get_width() / 2, height_ind),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=fontsize_annotations,
                            fontweight=fontweight_annotation,
                            color="#333333",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    alpha=0.8, edgecolor="none"),
                        )

                    if annotate_maj:
                        ax.annotate(
                            f"{height_maj*100:.1f}%",
                            xy=(bar_maj.get_x() + bar_maj.get_width() / 2, height_maj),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=fontsize_annotations,
                            fontweight=fontweight_annotation,
                            color="#333333",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    alpha=0.8, edgecolor="none"),
                        )

            current_offset += 2 * bar_width + group_spacing

        # --- styling ---
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

        ax.grid(True, linestyle="-", alpha=0.3, axis="y", linewidth=0.8)
        ax.grid(True, linestyle=":", alpha=0.2, axis="x", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [label.replace("_", " ").title() for label in labels],
            rotation=25, ha=font_x_orientation,
            fontsize=fontsize_x, fontweight="normal",
        )
        ax.set_ylabel(label_y, fontsize=fontsize_y, fontweight="bold", color="#333333")

        # y-limits with errors
        max_with_error = max_val
        min_with_error = min_val
        if show_error_bars:
            for data in chunk_dict.values():
                max_with_error = max(max_with_error,
                    max([data["individual"][i] + data["std"][i] 
                         for i in range(len(data["individual"]))]))
                min_with_error = min(min_with_error,
                    min([data["individual"][i] - data["std"][i] 
                         for i in range(len(data["individual"]))]))

        ax.set_ylim(min_with_error * 0.95, max_with_error * 1.05)

        def percentage_formatter(x, pos):
            return f"{x*100:.0f}%"

        ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        ax.tick_params(axis="y", labelsize=fontsize_y, colors="#333333")
        ax.tick_params(axis="x", labelsize=fontsize_x, colors="#333333")

        # Subplot title
        subplot_title = f"{title} (Part {subplot_idx + 1}/{n_subplots})" if n_subplots > 1 else title
        if subplot_title:
            ax.set_title(subplot_title, fontsize=title_fontsize, 
                        fontweight="bold", pad=25, color="#333333")

        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1),
            fontsize=10, frameon=True, fancybox=True,
            shadow=True, framealpha=0.95,
            edgecolor="#333333", facecolor="white",
        )

        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    return fig, axes


# Groups (same semantics as your plotting code)
groups_dict_not_reduced = {
    "all_items": [0,1,2,3,-99],
    #"positive_items": [1,2,3],
    #"keine_Aussage": [-99],
    "positive_items_and_kA": [1,2,3,-99],
}
groups_dict_reduced = {
    "all_items": [0,1,-99],
    #"positive_items": [1],
    #"keine_Aussage": [-99],
    "positive_items_and_kA": [1,-99],
}

def extract_psy_cols(df, psy_cols_regex=None):
    if psy_cols_regex is None:
        psy_cols_regex = r'^p\d+_.*_final$'
    return [c for c in df.columns if re.match(psy_cols_regex, c)]

def prepare_reference_rows(reference_df, ref_ID_col, vid_ids_to_use):
    """
    reference_df: long-ish with one row per video.
    Returns: dict[video_id] -> 1xN DataFrame of reference ratings (mapped with map_ref_rating).
    """
    ref = reference_df.copy()
    psy_cols = extract_psy_cols(ref)
    out = {}
    for vid, sub in ref.groupby(ref_ID_col):
        if vid_ids_to_use is not None and vid not in vid_ids_to_use:
            continue
        # each sub should be 1 row
        row_vals = sub.iloc[0][psy_cols]
        out[vid] = pd.DataFrame([row_vals.values], columns=psy_cols)
    return out

def split_master_to_site_matrices(master_df, rat_VID_col="ID_Video", SITE_COL="site", RATER_ID_COL="ids"
                                  , psy_cols_regex=None):
    """
    master_df: contains columns [SITE_COL, RATER_ID_COL, VID_COL] + psy_cols (wide ratings).
    Returns: dict[video_id][site_key] = DataFrame with columns psy_cols + [RATER_ID_COL]
    """
    psy_cols = extract_psy_cols(master_df, psy_cols_regex=psy_cols_regex)
    result = {}
    for vid, dfv in master_df.groupby(rat_VID_col):
        result[vid] = {}
        for site, dfs in dfv.groupby(SITE_COL):
            # keep only rating columns + rater id
            sub = dfs[psy_cols + [RATER_ID_COL]].copy()
            result[vid][site] = sub
    return result

def compute_accuracy(
    rating_ref, rating_df, id_col="ids"
):
    try:
        # 1) align and extract reference
        if id_col is not None:
            rating_df = rating_df.set_index(id_col)
        if isinstance(rating_ref, pd.DataFrame):
            if rating_ref.shape[0] != 1:
                raise ValueError("rating_ref DataFrame must have exactly one row")
            rating_ref = rating_ref.iloc[0]
        if not rating_df.columns.equals(rating_ref.index):
            raise ValueError("Columns of rating_df must match index of rating_ref")

        # 2) build a mask of “valid comparisons” (where neither is NaN)
        valid = rating_df.notna() & rating_ref.notna()

        # 3) boolean matrix of correctness *only where valid*
        correct = rating_df.eq(rating_ref, axis=1) & valid
        if correct.empty:
            print("Warning: No valid ratings found.")
            
        # 4) Per‐rater accuracy = correct_count / n_rated_items
        n_rated = valid.sum(axis=1)                       # how many items each rater actually rated
        correct_count = correct.sum(axis=1)
        accuracy = correct_count / n_rated
        mean_accuracy = accuracy.mean()

        # 5) Balanced accuracy per rater (filter out NaNs before score)
        def row_balanced(row, ref):
            mask = row.notna() & ref.notna()
            if mask.sum() == 0:
                return float('nan')
            return 0.7
        balanced_accuracies = rating_df.apply(lambda row: row_balanced(row, rating_ref), axis=1)
        mean_balanced_accuracy = balanced_accuracies.mean()

        # 6) Per‐item summary: only count raters who rated that item
        n_raters_per_item = valid.sum(axis=0)
        n_errors    = (valid & ~rating_df.eq(rating_ref, axis=1)).sum(axis=0)
        error_rate  = n_errors / n_raters_per_item

        item_summary = pd.DataFrame({
            'ref_rating':       rating_ref,
            'n_raters':         n_raters_per_item,
            'n_errors':         n_errors,
            'error_rate':       error_rate
        })

        # 7) rating distributions (only over those who rated)
        counts = (
            rating_df.where(valid)                   # mask out NaNs
            .apply(pd.Series.value_counts)           # per-item counts
            .fillna(0).astype(int)
            .T
        )
        counts.columns = [f'count_{val}' for val in counts.columns]
        item_summary = pd.concat([item_summary, counts], axis=1)

        # 8) who was wrong on each item
        raters_wrong_df = (
            (valid & ~rating_df.eq(rating_ref, axis=1))
            .apply(lambda col: list(col[col].index), axis=0)
        )
        # Convert to Series properly
        if isinstance(raters_wrong_df, pd.DataFrame):
            if raters_wrong_df.empty:
                raters_wrong = pd.Series([[] for _ in range(rating_df.shape[1])], index=rating_df.columns, name='raters_wrong')
            else:
                raters_wrong = raters_wrong_df.squeeze().rename('raters_wrong')
        else:
            if raters_wrong_df.empty:  # Works for Series too
                raters_wrong = pd.Series([[] for _ in range(rating_df.shape[1])], index=rating_df.columns, name='raters_wrong')
            else:
                raters_wrong = raters_wrong_df.rename('raters_wrong')
        
        item_summary = pd.concat([item_summary, raters_wrong], axis=1)
        raters_correct_df = correct.apply(lambda col: list(col[col].index), axis=0)
        #check if df is empty
        
        if isinstance(raters_correct_df, pd.DataFrame):
            if raters_correct_df.empty:
                raters_correct = pd.Series([[] for _ in range(rating_df.shape[1])], index=rating_df.columns, name='raters_correct')
            else:
                raters_correct = raters_correct_df.squeeze().rename('raters_correct')
        else:
            if raters_correct_df.empty:  # Works for Series too
                raters_correct = pd.Series([[] for _ in range(rating_df.shape[1])], index=rating_df.columns, name='raters_correct')
            else:
                raters_correct = raters_correct_df.rename('raters_correct')

        # how many got it right
        n_correct = correct.sum(axis=0).rename('n_correct')

        # correct rate (out of raters who rated)
        correct_rate = (n_correct / n_raters_per_item).rename('correct_rate')

        # 10) CONCATENATE into your existing summary
        item_summary = pd.concat([
            item_summary,
            n_correct,
            correct_rate,
            raters_correct
        ], axis=1)

        return accuracy, mean_accuracy, balanced_accuracies, mean_balanced_accuracy, item_summary
    except Exception as e:
        print(f"Error in compute_accuracy: {e}")
        raise   

def compute_multiple_accuracies_for_filtered_rating_ranges(
    rating_ref: pd.DataFrame,
    rating_df: pd.DataFrame,
    groups_dict: dict[str, list[int]],
    id_col: str = "ids",
):
    try:
        results = {}
        for group_name, allowed in groups_dict.items():
            # make copies so we don’t clobber originals
            ref_row = rating_ref.iloc[0]
            keep_cols = ref_row.index[ref_row.isin(allowed)]
            
            # 2) subset both reference and ratings to those columns
            ref_sub = rating_ref[keep_cols]
            if id_col and id_col in rating_df.columns:
                df_sub = rating_df[keep_cols.tolist() + [id_col]].copy()
            else:
                df_sub = rating_df[keep_cols].copy()
            
            # call your existing function
            out = compute_accuracy(
                ref_sub,
                df_sub,
                id_col=id_col
            )
            results[group_name] = out
            #print(f"Computed accuracy for group '{group_name}' with {len(keep_cols)} items.")
            #print(f"Mean accuracy: {out[1]:.4f}")
    except Exception as e:
        print(f"Error computing accuracy for group '{group_name}': {e}")
        raise
    return results

def compute_site_results_for_video(site_frames, ref_row, psy_cols, RATER_ID_COL="ids"):
    """
    site_frames: dict[site_key] -> df with columns psy_cols + [ID_COL]
    ref_row: 1xN DataFrame with columns psy_cols
    Returns: two dicts (not_reduced_data, reduced_data) and same for mode.
      Each is shaped site_key -> group_name -> tuple from compute_accuracy().
    """
    # reduce to 3 categories
    all_dfs = [site_frames[k] for k in site_frames] + [ref_row]
    reduced = reduce_data_to_3_categories(psy_cols, all_dfs)
    site_frames_reduced = {k: reduced[i] for i, k in enumerate(site_frames.keys())}
    ref_reduced = reduced[-1]
    reduced_psy_cols = [c + "_reduced" for c in psy_cols]

    not_reduced_data = {}
    reduced_data = {}
    not_reduced_mode = {}
    reduced_mode = {}

    for site_key, site_df in site_frames.items():
        print(f"Computeing site '{site_key}'")
        # Individual accuracies
        nr = compute_multiple_accuracies_for_filtered_rating_ranges(
            ref_row[psy_cols],
            site_df[psy_cols + [RATER_ID_COL]],
            groups_dict_not_reduced,
            id_col=RATER_ID_COL,
        )
        rd = compute_multiple_accuracies_for_filtered_rating_ranges(
            ref_reduced[reduced_psy_cols],
            site_frames_reduced[site_key][reduced_psy_cols + [RATER_ID_COL]],
            groups_dict_reduced,
            id_col=RATER_ID_COL,
        )

        # Majority vote row (mode)
        mode_row = site_df[psy_cols].mode(axis=0).iloc[0].to_frame().T
        mode_row[RATER_ID_COL] = 0
        mode_row_reduced = reduce_data_to_3_categories(psy_cols, [mode_row])[0]

        nr_mode = compute_multiple_accuracies_for_filtered_rating_ranges(
            ref_row[psy_cols],
            mode_row[psy_cols + [RATER_ID_COL]],
            groups_dict_not_reduced,
            id_col=RATER_ID_COL,
        )
        rd_mode = compute_multiple_accuracies_for_filtered_rating_ranges(
            ref_reduced[reduced_psy_cols],
            mode_row_reduced[reduced_psy_cols + [RATER_ID_COL]],
            groups_dict_reduced,
            id_col=RATER_ID_COL,
        )

        not_reduced_data[site_key] = nr
        reduced_data[site_key] = rd
        not_reduced_mode[site_key] = nr_mode
        reduced_mode[site_key] = rd_mode

    return (not_reduced_data, reduced_data, not_reduced_mode, reduced_mode)

def aggregate_over_videos(per_video_results):
    """
    per_video_results: dict[vid] -> (not_red, red, mode_not_red, mode_red)
      where each 'not_red' etc is site_key -> group_name -> tuple from compute_accuracy
    Returns aggregated dicts in the same shape expected by plot_multi_site_accuracies.
    We concatenate per-rater accuracy Series across videos WITHOUT grouping by rater ID
    since different raters rated different videos.
    """
    def agg(blocks):
        # blocks: list of tuples (acc_series, mean, bal_series, bal_mean, item_summary) per video
        if not blocks:
            return None
        
        # Simply concatenate all accuracy values across videos
        # Each rater's accuracy for each video is treated as independent observation
        acc_series = pd.concat([b[0] for b in blocks], ignore_index=True)
        mean_acc = float(acc_series.mean())
        
        if blocks[0][2] is not None:
            # Same for balanced accuracy - just concatenate
            bal = pd.concat([b[2] for b in blocks], ignore_index=True)
            mean_bal = float(bal.mean())
        else:
            bal = None
            mean_bal = np.nan
        
        return (acc_series, mean_acc, bal, mean_bal, None)

    # collect site keys, group names from the first video
    vids = list(per_video_results.keys())
    if not vids:
        return {}, {}, {}, {}

    # Determine sites and groups robustly
    any_not_red = per_video_results[vids[0]][0]
    sites = list(any_not_red.keys())
    groups_nr = list(next(iter(any_not_red.values())).keys())

    aggregated_not_reduced = {s: {} for s in sites}
    aggregated_reduced = {s: {} for s in sites}
    aggregated_mode_not_red = {s: {} for s in sites}
    aggregated_mode_reduced = {s: {} for s in sites}

    for site in sites:
        for g in groups_nr:
            nr_blocks = []
            rd_blocks = []
            nr_mode_blocks = []
            rd_mode_blocks = []
            for vid in vids:
                nr, rd, nr_mode, rd_mode = per_video_results[vid]
                if site in nr and g in nr[site]:
                    nr_blocks.append(nr[site][g])
                if site in rd and g in rd[site]:
                    rd_blocks.append(rd[site][g])
                if site in nr_mode and g in nr_mode[site]:
                    nr_mode_blocks.append(nr_mode[site][g])
                if site in rd_mode and g in rd_mode[site]:
                    rd_mode_blocks.append(rd_mode[site][g])

            aggregated_not_reduced[site][g]   = agg(nr_blocks)
            aggregated_reduced[site][g]       = agg(rd_blocks)
            aggregated_mode_not_red[site][g]  = agg(nr_mode_blocks)
            aggregated_mode_reduced[site][g]  = agg(rd_mode_blocks)

    return aggregated_not_reduced, aggregated_reduced, aggregated_mode_not_red, aggregated_mode_reduced

# -------- WORKFLOW (A) all videos vs AI --------
def plot_all_videos_vs_ai(human_master, reference_all, site_config, title_suffix="All videos", 
                          psy_cols_regex=None,
                          vid_ids_to_use=[7,8,9],
                          ref_VID_col="ID_Video",
                          rat_VID_col="ID_Video",
                          SITE_COL="site",
                          RATER_ID_COL="ids",
                          show_annotations=True,
                          show_error_bars=False,
                          groups_dict_not_reduced=None,
                          groups_dict_reduced=None,
                          annotate_best_only=True,
                          fontsize_annotations=12,
                          show_std_values=True,
                          skip_plotting = False
                          ):
    
    if groups_dict_not_reduced is None:
        groups_dict_not_reduced = {
            "all_items": [0,1,2,3,-99],
            #"positive_items": [1,2,3],
            #"keine_Aussage": [-99],
            "positive_items_and_kA": [1,2,3,-99],
        }
    if groups_dict_reduced is None:
        groups_dict_reduced = {
            "all_items": [0,1,-99],
            #"positive_items": [1],
            #"keine_Aussage": [-99],
            "positive_items_and_kA": [1,-99],
        }
    psy_cols = extract_psy_cols(human_master, psy_cols_regex)
    ref_rows = prepare_reference_rows(reference_all, ref_ID_col=ref_VID_col, vid_ids_to_use=vid_ids_to_use)
    per_video_results = {}

    site_frames_all = split_master_to_site_matrices(human_master,rat_VID_col=rat_VID_col,
                                                    SITE_COL=SITE_COL, RATER_ID_COL=RATER_ID_COL, psy_cols_regex=psy_cols_regex)
    for vid, site_frames in site_frames_all.items():
        if vid not in ref_rows:
            continue
        nr, rd, nr_mode, rd_mode = compute_site_results_for_video(site_frames, ref_rows[vid], psy_cols, RATER_ID_COL=RATER_ID_COL)
        per_video_results[vid] = (nr, rd, nr_mode, rd_mode)

    agg_nr, agg_rd, agg_nr_mode, agg_rd_mode = aggregate_over_videos(per_video_results)
    if skip_plotting:
        return None, None, (agg_nr, agg_rd, agg_nr_mode, agg_rd_mode)
    # Use your existing plotter
    fig1, ax1 = plot_multi_site_accuracies_split(
        groups_dict_not_reduced,
        agg_nr,
        agg_nr_mode,
        site_config,
        index=1,
        title=f"Accuracies — Not Reduced ({title_suffix})",
        show_annotations=show_annotations,
        show_error_bars=show_error_bars,
        annotate_best_only=annotate_best_only,
        fontsize_annotations=fontsize_annotations,
        show_std_values=show_std_values
    )
    fig2, ax2 = plot_multi_site_accuracies_split(
        groups_dict_reduced,
        agg_rd,
        agg_rd_mode,
        site_config,
        index=1,
        title=f"Accuracies — Reduced ({title_suffix})",
        show_annotations=show_annotations,
        show_error_bars=show_error_bars,
        annotate_best_only=annotate_best_only,
        fontsize_annotations=fontsize_annotations,
        show_std_values=show_std_values
    )
    return (fig1, ax1), (fig2, ax2), (agg_nr, agg_rd, agg_nr_mode, agg_rd_mode)

def _parse_model_key(key: str):
    temp = 0.5
    knowledge = None
    base = key
    m = re.search(r"_temp_(\d+(?:\.\d+)?)_([0-9]+)$", key)
    if m:
        temp = float(m.group(1))
        knowledge = int(m.group(2))
        base = key[: m.start()]
    else:
        m2 = re.search(r"_(\d+)$", key)
        if m2:
            knowledge = int(m2.group(1))
            base = key[: m2.start()]
    return base, temp, knowledge


def _extract_series(entry):
    # entry is expected to be the value of dict['all_items'] -> tuple/list
    # The first element is the per-video values (Series-like)
    if not isinstance(entry, (tuple, list)) or len(entry) == 0:
        return None
    s = entry[0]
    try:
        s = pd.Series(s).astype(float).dropna()
    except Exception:
        try:
            s = s.astype(float).dropna()
        except Exception:
            return None
    return s


def _fmt_stats(series: pd.Series, range_option="sd"):
    if series is None or len(series) == 0:
        return None, None, ""
    m = float(np.mean(series))
    if range_option == "sd":
        sd = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
        return m, sd, f"{m:.2f} ± {sd:.2f}"
    elif range_option == "range":
        range_min = float(np.min(series))
        range_max = float(np.max(series))
        return m, (range_min, range_max), f"{m:.2f} ({range_min:.2f} - {range_max:.2f})"
    else:
        raise ValueError(f"Unknown range_option: {range_option}")




def build_performance_table(agg: dict, range_option="sd") -> pd.DataFrame:
    # Build nested structure: base -> temp -> knowledge -> series
    content = {}
    for key, d in agg.items():
        if not isinstance(d, dict):
            continue
        if "all_items" not in d:
            continue
        s = _extract_series(d["all_items"])
        if s is None:
            continue
        base, temp, knowledge = _parse_model_key(key)
        if knowledge is None:
            continue
        content.setdefault(base, {}).setdefault(temp, {})[knowledge] = s

    records = []
    for base, temps in content.items():
        # Stats for temp=0.5
        s_t05_no = temps.get(0.5, {}).get(0)
        s_t05_yes = temps.get(0.5, {}).get(10)

        # Stats for temp=0.0
        s_t0_no = temps.get(0.0, {}).get(0)
        s_t0_yes = temps.get(0.0, {}).get(10)
        _, _, txt_t0_no = _fmt_stats(s_t0_no, range_option=range_option)
        _, _, txt_t0_yes = _fmt_stats(s_t0_yes, range_option=range_option)

        # Also explicitly list temp=0.5 combos to avoid ambiguity
        _, _, txt_t05_no = _fmt_stats(s_t05_no, range_option=range_option)
        _, _, txt_t05_yes = _fmt_stats(s_t05_yes, range_option=range_option)

        if range_option == "sd":
            measure_text = "(mean ± sd)"
        elif range_option == "range":
            measure_text = "(min - max)"
        records.append(
            {
                "Model": base,
                f"Temp 0 No Knowledge {measure_text}": txt_t0_no,
                f"Temp 0 With Knowledge {measure_text}": txt_t0_yes,
                f"Temp 0.5 No Knowledge {measure_text}": txt_t05_no,
                f"Temp 0.5 With Knowledge {measure_text}": txt_t05_yes,
            }
        )

    df = pd.DataFrame(records).sort_values("Model").reset_index(drop=True)
    return df

def build_merged_performance_table(
    df_nr: pd.DataFrame, 
    df_rd: pd.DataFrame, 
    sort_by: str = 'reduced_max',
    range_option: str = 'sd'
) -> pd.DataFrame:
    """
    Build a merged table with both not-reduced and reduced scores in the same cell.
    Each model gets columns showing: "not_reduced [range] / reduced [range]"
    
    Auto-detects column format from actual column names.
    """
    
    # Auto-detect range_option by checking actual column names
    sample_cols_nr = list(df_nr.columns)
    sample_cols_rd = list(df_rd.columns)
    all_cols = sample_cols_nr + sample_cols_rd
    
    # Check what format we have
    has_sd = any('mean ± SD' in str(col) or 'mean ± sd' in str(col).lower() for col in all_cols)
    has_range = any('min - max' in str(col) for col in all_cols)
    
    if has_range:
        detected_range_option = 'range'
        col_pattern = '(min - max)'
    elif has_sd:
        detected_range_option = 'sd'
        col_pattern = '(mean ± SD)'
    else:
        detected_range_option = range_option
        col_pattern = '(min - max)' if range_option == 'range' else '(mean ± SD)'
    
    print(f"Detected range_option: {detected_range_option}")
    print(f"Looking for pattern: '{col_pattern}'")
    
    # Find all condition columns (they all contain the pattern) - BOTH _NR and _R versions
    condition_cols_nr = [c for c in df_nr.columns if col_pattern in str(c) and c != 'Model']
    condition_cols_rd = [c for c in df_rd.columns if col_pattern in str(c) and c != 'Model']
    
    print(f"\nFound {len(condition_cols_nr)} condition columns in not-reduced")
    print(f"Found {len(condition_cols_rd)} condition columns in reduced")
    
    if len(condition_cols_nr) == 0 or len(condition_cols_rd) == 0:
        raise ValueError(
            f"No matching columns found!\n"
            f"Not-reduced columns: {list(df_nr.columns)}\n"
            f"Reduced columns: {list(df_rd.columns)}\n"
            f"Looking for pattern: '{col_pattern}'"
        )
    
    # Merge on Model name
    df_merged = df_nr.merge(
        df_rd,
        on='Model',
        how='outer',
        suffixes=('_NR', '_R')
    )
    
    print(f"Merged dataframe shape: {df_merged.shape}")
    print(f"Merged columns: {list(df_merged.columns)}")
    
    # Function to combine two score strings
    def combine_scores(nr_text, rd_text):
        if pd.isna(nr_text) and pd.isna(rd_text):
            return ""
        elif pd.isna(nr_text):
            return f"- / {rd_text}"
        elif pd.isna(rd_text):
            return f"{nr_text} / -"
        else:
            return f"{nr_text} / {rd_text}"
    
    # Create result DataFrame
    result_df = pd.DataFrame()
    result_df['Model'] = df_merged['Model']
    
    # Extract base condition names by removing the _NR and _R suffixes
    # and the pattern itself to get unique condition names
    base_conditions = set()
    for col in df_merged.columns:
        if col == 'Model':
            continue
        # Remove _NR or _R suffix
        base = col.replace('_NR', '').replace('_R', '')
        # Remove the pattern to get the condition name
        base = base.replace(col_pattern, '').strip()
        base_conditions.add(base)
    
    base_conditions = sorted(base_conditions)
    
    # Process each condition
    for base_cond in base_conditions:
        # Find the corresponding _NR and _R columns
        nr_col_name = f"{base_cond} {col_pattern}_NR"
        rd_col_name = f"{base_cond} {col_pattern}_R"
        
        print(f"\nLooking for NR column: '{nr_col_name}'")
        print(f"Looking for R column: '{rd_col_name}'")
        print(f"Available columns: {[c for c in df_merged.columns if base_cond in c]}")
        
        if nr_col_name not in df_merged.columns or rd_col_name not in df_merged.columns:
            print(f"Warning: Skipping {base_cond} (columns not found)")
            continue
        
        # Create new column name
        if detected_range_option == 'sd':
            new_col_name = f"{base_cond} (mean ± SD) (NR / R)"
        else:
            new_col_name = f"{base_cond} (min - max) (NR / R)"
        
        result_df[new_col_name] = df_merged.apply(
            lambda row: combine_scores(row[nr_col_name], row[rd_col_name]),
            axis=1
        )
    
    # Helper function to extract numeric value for sorting
    def extract_numeric_value(val_str, part_idx=0):
        """Extract numeric value from 'X.XX ± X.XX' or 'X.XX (X.XX - X.XX)' format"""
        if pd.isna(val_str) or val_str == "":
            return -np.inf
        if " / " in val_str:
            parts = val_str.split(" / ")
            val_str = parts[part_idx] if part_idx < len(parts) else parts[0]
        
        if val_str == "-":
            return -np.inf
        
        # Extract first numeric value (the mean)
        try:
            if "±" in val_str:
                mean_val = val_str.split("±")[0].strip()
            elif "(" in val_str:
                mean_val = val_str.split("(")[0].strip()
            else:
                mean_val = val_str
            return float(mean_val)
        except Exception as e:
            print(f"Warning: Could not parse '{val_str}': {e}")
            return -np.inf
    
    # Sort the dataframe
    if sort_by == 'reduced_max':
        def get_max_reduced_acc(row):
            max_val = -np.inf
            for col in result_df.columns:
                if col == 'Model':
                    continue
                val_str = row[col]
                # Extract reduced value (second part after " / ")
                numeric_val = extract_numeric_value(val_str, part_idx=1)
                max_val = max(max_val, numeric_val)
            return max_val
        
        result_df['__sort_key__'] = result_df.apply(get_max_reduced_acc, axis=1)
        result_df = result_df.sort_values('__sort_key__', ascending=False).drop('__sort_key__', axis=1)
        
    elif sort_by == 'not_reduced_max':
        def get_max_not_reduced_acc(row):
            max_val = -np.inf
            for col in result_df.columns:
                if col == 'Model':
                    continue
                val_str = row[col]
                # Extract not-reduced value (first part before " / ")
                numeric_val = extract_numeric_value(val_str, part_idx=0)
                max_val = max(max_val, numeric_val)
            return max_val
        
        result_df['__sort_key__'] = result_df.apply(get_max_not_reduced_acc, axis=1)
        result_df = result_df.sort_values('__sort_key__', ascending=False).drop('__sort_key__', axis=1)
        
    elif sort_by == 'model':
        result_df = result_df.sort_values('Model')
    
    result_df = result_df.reset_index(drop=True)
    
    return result_df



def compute_per_video_results_reduced(
    human_master,
    reference_all,
    vid_ids=(7, 8, 9),
    ref_VID_col="ID_Video",
    rat_VID_col="video_id",
    SITE_COL="site",
    RATER_ID_COL="ids",
    psy_cols_regex=None,
):
    psy_cols = extract_psy_cols(human_master, psy_cols_regex)
    ref_rows = prepare_reference_rows(reference_all, ref_ID_col=ref_VID_col, vid_ids_to_use=list(vid_ids))
    site_frames_all = split_master_to_site_matrices(
        human_master,
        rat_VID_col=rat_VID_col,
        SITE_COL=SITE_COL,
        RATER_ID_COL=RATER_ID_COL,
        psy_cols_regex=psy_cols_regex,
    )
    per_video_results = {}
    for vid, site_frames in site_frames_all.items():
        if vid not in ref_rows:
            continue
        nr, rd, nr_mode, rd_mode = compute_site_results_for_video(
            site_frames,
            ref_rows[vid],
            psy_cols,
            RATER_ID_COL=RATER_ID_COL,
        )
        per_video_results[vid] = (nr, rd, nr_mode, rd_mode)
    return per_video_results


def build_best_per_video_table(per_video_results, use_reduced=True, video_labels={7: "Mania", 8: "Depression", 9: "Schizophrenia"}):
    idx = 1 if use_reduced else 0  # 1 => reduced, 0 => not reduced
    rows = {}
    for vid, blocks in per_video_results.items():
        if vid not in video_labels:
            continue
        rd = blocks[idx]
        for model_key, group_dict in rd.items():
            if "all_items" not in group_dict:
                continue
            entry = group_dict["all_items"]
            # entry: (acc_series, mean_acc, bal_series, bal_mean, None)
            s = pd.Series(entry[0]).astype(float).dropna()
            mean = s.mean()
            sd = s.std(ddof=1) if len(s) > 1 else 0.0
            base, _, _ = _parse_model_key(model_key)
            rows.setdefault(base, {})[vid] = (mean, sd)

    records = []
    for base, vid_stats in rows.items():
        rec = {"Model": base}
        for vid, label in video_labels.items():
            m, sd = vid_stats.get(vid, (np.nan, np.nan))
            rec[label] = "" if np.isnan(m) else f"{m:.2f}"
        records.append(rec)

    df = pd.DataFrame(records).sort_values("Model").reset_index(drop=True)
    return df

def _fmt_mean_sd(mean, sd):
    if np.isnan(mean):
        return ""
    return f"{mean:.3f}" if (np.isnan(sd) or sd == 0) else f"{mean:.3f} ± {sd:.3f}"

def _fmt_mean_range(mean, range_tuple):
    if np.isnan(mean):
        return ""
    if range_tuple is None or any(np.isnan(x) for x in range_tuple):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({range_tuple[0]:.3f} - {range_tuple[1]:.3f})"

def _extract_mean_sd(entry):
    if "all_items" not in entry:
        return (np.nan, np.nan)
    s = pd.Series(entry["all_items"][0]).astype(float).dropna()
    mean = s.mean()
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    return mean, sd

def _extract_mean_range(entry):
    if "all_items" not in entry:
        return (np.nan, None)
    s = pd.Series(entry["all_items"][0]).astype(float).dropna()
    mean = s.mean()
    if len(s) > 1:
        range_tuple = (s.min(), s.max())
    else:
        range_tuple = None
    return mean, range_tuple

def _get_block(per_video_results, vid, use_reduced: bool):
    idx = 1 if use_reduced else 0
    blocks = per_video_results.get(vid, None)
    if blocks is None:
        return None
    return blocks[idx]  # dict[model_key] -> group_dict

def build_best_per_video_table_with_settings(per_video_results, use_reduced=True, combine_nr_r=False, use_range=False, video_labels={7: "Mania", 8: "Depression", 9: "Schizophrenia"}):
    """
    When combine_nr_r=True, each cell shows: 'NR / R' (means ± sd or with ranges).
    When use_range=True, shows (min - max) instead of ± sd.
    For T0 models: only show mean without sd/range.
    """
    rows = {}
    for vid in per_video_results:
        if vid not in video_labels:
            continue
        block_r = _get_block(per_video_results, vid, use_reduced=True)
        block_nr = _get_block(per_video_results, vid, use_reduced=False)
        block_use = block_r if use_reduced else block_nr
        if block_use is None:
            continue

        for model_key, group_dict in block_use.items():
            if "all_items" not in group_dict:
                continue

            base, temp, knowledge = _parse_model_key(model_key)
            knowledge_str = "With-K" if knowledge == 10 else "No-K"
            temp_str = f"T{temp:.1f}".replace(".0", "")
            model_name = f"{base} [{temp_str}-{knowledge_str}]"

            # Check if T0 model
            is_t0 = (temp == 0.0)

            if use_range:
                m_use, range_use = _extract_mean_range(group_dict)
                if is_t0:
                    cell = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell = _fmt_mean_range(m_use, range_use)
            else:
                m_use, sd_use = _extract_mean_sd(group_dict)
                if is_t0:
                    cell = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell = _fmt_mean_sd(m_use, sd_use)

            if combine_nr_r:
                # fetch both NR and R
                if use_range:
                    nr_mean, nr_range = _extract_mean_range(block_nr.get(model_key, {})) if block_nr else (np.nan, None)
                    r_mean, r_range = _extract_mean_range(block_r.get(model_key, {})) if block_r else (np.nan, None)
                    if is_t0:
                        cell = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell = f"{_fmt_mean_range(nr_mean, nr_range)} / {_fmt_mean_range(r_mean, r_range)}"
                else:
                    nr_mean, nr_sd = _extract_mean_sd(block_nr.get(model_key, {})) if block_nr else (np.nan, np.nan)
                    r_mean, r_sd = _extract_mean_sd(block_r.get(model_key, {})) if block_r else (np.nan, np.nan)
                    if is_t0:
                        cell = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell = f"{_fmt_mean_sd(nr_mean, nr_sd)} / {_fmt_mean_sd(r_mean, r_sd)}"

            rows.setdefault(model_name, {})[vid] = cell

    records = []
    for model_name, vid_stats in rows.items():
        rec = {"Model": model_name}
        for vid, label in video_labels.items():
            rec[label] = vid_stats.get(vid, "")
        records.append(rec)

    return pd.DataFrame(records).sort_values("Model").reset_index(drop=True)

def build_best_per_video_table_compact(per_video_results, use_reduced=True, combine_nr_r=False, use_range=False, video_labels={7: "Mania", 8: "Depression", 9: "Schizophrenia"}):
    rows = {}
    for vid in per_video_results:
        if vid not in video_labels:
            continue
        block_r = _get_block(per_video_results, vid, use_reduced=True)
        block_nr = _get_block(per_video_results, vid, use_reduced=False)
        block_use = block_r if use_reduced else block_nr
        if block_use is None:
            continue

        for model_key, group_dict in block_use.items():
            if "all_items" not in group_dict:
                continue
            base, temp, knowledge = _parse_model_key(model_key)
            temp_suffix = "T0" if temp == 0.0 else "T05" if temp == 0.5 else f"T{temp}"
            know_suffix = "K" if knowledge == 10 else "NK"
            model_name = f"{base}_{temp_suffix}_{know_suffix}"

            # Check if T0 model
            is_t0 = (temp == 0.0)

            if use_range:
                m_use, range_use = _extract_mean_range(group_dict)
                if is_t0:
                    cell = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell = _fmt_mean_range(m_use, range_use)
            else:
                m_use, sd_use = _extract_mean_sd(group_dict)
                if is_t0:
                    cell = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell = _fmt_mean_sd(m_use, sd_use)

            if combine_nr_r:
                if use_range:
                    nr_mean, nr_range = _extract_mean_range(block_nr.get(model_key, {})) if block_nr else (np.nan, None)
                    r_mean, r_range = _extract_mean_range(block_r.get(model_key, {})) if block_r else (np.nan, None)
                    if is_t0:
                        cell = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell = f"{_fmt_mean_range(nr_mean, nr_range)} / {_fmt_mean_range(r_mean, r_range)}"
                else:
                    nr_mean, nr_sd = _extract_mean_sd(block_nr.get(model_key, {})) if block_nr else (np.nan, np.nan)
                    r_mean, r_sd = _extract_mean_sd(block_r.get(model_key, {})) if block_r else (np.nan, np.nan)
                    if is_t0:
                        cell = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell = f"{_fmt_mean_sd(nr_mean, nr_sd)} / {_fmt_mean_sd(r_mean, r_sd)}"

            rows.setdefault(model_name, {})[vid] = cell

    records = []
    for model_name, vid_stats in rows.items():
        rec = {"Model": model_name}
        for vid, label in video_labels.items():
            rec[label] = vid_stats.get(vid, "")
        records.append(rec)

    return pd.DataFrame(records).sort_values("Model").reset_index(drop=True)

def build_best_per_video_with_summary_stats(per_video_results, use_reduced=True, combine_nr_r=False, use_range=False, video_labels={7: "Mania", 8: "Depression", 9: "Schizophrenia"}):
    rows = {}
    for vid in per_video_results:
        if vid not in video_labels:
            continue
        block_r = _get_block(per_video_results, vid, use_reduced=True)
        block_nr = _get_block(per_video_results, vid, use_reduced=False)
        block_use = block_r if use_reduced else block_nr
        if block_use is None:
            continue

        for model_key, group_dict in block_use.items():
            if "all_items" not in group_dict:
                continue
            base, temp, knowledge = _parse_model_key(model_key)
            knowledge_str = "With-K" if knowledge == 10 else "No-K"
            temp_str = f"T{temp:.1f}".replace(".0", "")
            model_name = f"{base} [{temp_str}-{knowledge_str}]"

            # Check if T0 model
            is_t0 = (temp == 0.0)

            if use_range:
                m_use, range_use = _extract_mean_range(group_dict)
                if is_t0:
                    cell_val = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell_val = _fmt_mean_range(m_use, range_use)
            else:
                m_use, sd_use = _extract_mean_sd(group_dict)
                if is_t0:
                    cell_val = f"{m_use:.3f}" if not np.isnan(m_use) else ""
                else:
                    cell_val = _fmt_mean_sd(m_use, sd_use)

            if combine_nr_r:
                if use_range:
                    nr_mean, nr_range = _extract_mean_range(block_nr.get(model_key, {})) if block_nr else (np.nan, None)
                    r_mean, r_range = _extract_mean_range(block_r.get(model_key, {})) if block_r else (np.nan, None)
                    if is_t0:
                        cell_val = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell_val = f"{_fmt_mean_range(nr_mean, nr_range)} / {_fmt_mean_range(r_mean, r_range)}"
                    summary_mean = np.nanmean([r_mean, nr_mean]) if not np.isnan(r_mean) else nr_mean
                else:
                    nr_mean, nr_sd = _extract_mean_sd(block_nr.get(model_key, {})) if block_nr else (np.nan, np.nan)
                    r_mean, r_sd = _extract_mean_sd(block_r.get(model_key, {})) if block_r else (np.nan, np.nan)
                    if is_t0:
                        cell_val = f"{nr_mean:.3f} / {r_mean:.3f}" if (not np.isnan(nr_mean) and not np.isnan(r_mean)) else ""
                    else:
                        cell_val = f"{_fmt_mean_sd(nr_mean, nr_sd)} / {_fmt_mean_sd(r_mean, r_sd)}"
                    summary_mean = np.nanmean([r_mean, nr_mean]) if not np.isnan(r_mean) else nr_mean
            else:
                summary_mean = m_use

            rows.setdefault(model_name, {}).setdefault("per_vid", {})[vid] = cell_val
            rows[model_name].setdefault("means_collect", []).append(summary_mean)

    records = []
    for model_name, info in rows.items():
        vid_stats = info["per_vid"]
        means_collect = pd.Series(info["means_collect"]).dropna()
        rec = {"Model": model_name}
        for vid, label in video_labels.items():
            rec[label] = vid_stats.get(vid, "")
        if not means_collect.empty:
            rec["Mean"] = f"{means_collect.mean():.3f}"
            rec["Min"]  = f"{means_collect.min():.3f}"
            rec["Max"]  = f"{means_collect.max():.3f}"
            rec["Range"] = f"{means_collect.max() - means_collect.min():.3f}"
        else:
            rec["Mean"] = rec["Min"] = rec["Max"] = rec["Range"] = ""
        records.append(rec)

    return pd.DataFrame(records).sort_values("Model").reset_index(drop=True)






def _wrap_hover(text: str, width: int = 60) -> str:
    """Wrap long text for Plotly hover by inserting <br> breaks."""
    if not isinstance(text, str) or not text:
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))

def create_interactive_error_scatter_with_begruendung(
    master_df: pd.DataFrame,
    reference: pd.DataFrame,
    best_ai: str,
    psy_cols: list,
    groups_dict_reduced: dict = None,
    video_ids: list = [7, 8, 9],
    jitter_amount: float = 30,
    base_title: str = "Error Rate Comparison: LLM vs. Humans",
    path_html: str = "../outputs/figs/",
    prefix: str = "",
    color_by_rating: bool = True,  # NEW: color by reference rating type
    show_threshold_lines: bool = True,  # NEW: show difficulty threshold lines
    low_error_threshold: float = 20.0,  # NEW: threshold for low difficulty
    high_error_threshold: float = 80.0,  # NEW: threshold for high difficulty
):
    """
    Create interactive scatter plots showing LLM vs Human error rates with LLM justifications.

    For each video:
    - X-axis: Human error rate (0-100%)
    - Y-axis: LLM error rate (0-100%)
    - Hover: Shows item name, error rates, reference rating, and LLM justification
    
    Parameters:
    -----------
    color_by_rating : bool
        If True, color points by reference rating type (0, 1, -99)
        If False, color by LLM correct/wrong (original behavior)
    show_threshold_lines : bool
        If True, show vertical threshold lines at low_error_threshold and high_error_threshold
    low_error_threshold : float
        Threshold below which items are considered "low difficulty" for humans
    high_error_threshold : float
        Threshold above which items are considered "high difficulty" for humans
    """
    if groups_dict_reduced is None:
        groups_dict_reduced = {
            "all_items": [0, 1, -99],
            "positive_items_and_kA": [1, -99],
        }
    
    # English video names
    video_names = {7: 'Mania', 8: 'Depression', 9: 'Schizophrenia'}
    
    # Rating type colors and labels (matching the quadrant plot)
    rating_colors = {
        0: '#3498db',    # blue for absent
        1: "#e87d2e",    # orange for present
        -99: '#95a5a6'   # gray for not assessable
    }
    rating_labels = {
        0: 'Absent (0)',
        1: 'Present (1)',
        -99: 'Not assessable (-99)'
    }

    figures = []
    plot_data_dict = {}
    
    for vid in video_ids:
        # Filter to single video
        master_vid = master_df[master_df['video_id'] == vid].copy()
        ref_vid = reference[reference['ID_Video'] == vid].copy()

        # Human and LLM ratings for this video
        rating_humans_copy = master_vid[master_vid['site'].isin(['clinic_3', 'clinic_1', 'clinic_2'])].copy()
        rating_ai_all = master_vid[master_vid['site'] == best_ai][psy_cols].copy()
        rating_ref = ref_vid[psy_cols].copy()

        # Get LLM justifications
        ai_begruendung_cols = [col.split('_')[0] + "_begründung" for col in psy_cols]
        rating_ai_begruendung = master_vid[master_vid['site'] == best_ai][ai_begruendung_cols].copy()

        # Reduced versions (3 categories)
        rating_ai_mode = rating_ai_all.mode(axis=0).iloc[0].to_frame().T
        rating_ai_mode['ids'] = 0

        all_dfs = [rating_humans_copy, rating_ai_mode, rating_ref, rating_ai_all]
        reduced_dfs = reduce_data_to_3_categories(psy_cols, all_dfs)

        rating_humans_copy_red = reduced_dfs[0]
        rating_ai_red = reduced_dfs[1]
        rating_ref_red = reduced_dfs[2]
        rating_ai_all_red = reduced_dfs[3]

        psy_cols_red = [c + "_reduced" for c in psy_cols]

        # Compute accuracies
        human_results = compute_multiple_accuracies_for_filtered_rating_ranges(
            rating_ref=rating_ref_red[psy_cols_red],
            rating_df=rating_humans_copy_red[psy_cols_red + ['ids']],
            groups_dict=groups_dict_reduced,
            id_col='ids'
        )
        ai_results = compute_multiple_accuracies_for_filtered_rating_ranges(
            rating_ref=rating_ref_red[psy_cols_red],
            rating_df=rating_ai_red[psy_cols_red + ['ids']],
            groups_dict=groups_dict_reduced,
            id_col='ids'
        )

        # Item summaries
        human_item_summary = human_results['all_items'][4]
        ai_item_summary = ai_results['all_items'][4]

        # Prepare data for plotting
        plot_data = []
        for item_red in psy_cols_red:
            item_base = item_red.replace('_reduced', '')
            begruendung_col = item_base.split('_')[0] + "_begründung"

            # Error rates
            human_error_rate = human_item_summary.loc[item_red, 'error_rate'] * 100
            ai_error_rate = ai_item_summary.loc[item_red, 'error_rate'] * 100

            # Reference and prediction (reduced)
            ref_rating = rating_ref_red[item_red].iloc[0]
            ai_prediction = rating_ai_red[item_red].iloc[0]

            # LLM justification
            maj_rating = rating_ai_red[item_red].iloc[0]
            matching_rows = rating_ai_all_red[rating_ai_all_red[item_red] == maj_rating]
            if not matching_rows.empty:
                first_match_idx = matching_rows.index[0]
                begruendung_text = rating_ai_begruendung.loc[first_match_idx, begruendung_col]
            else:
                print(f"had to use fallback for item {item_base}")
                begruendung_text = rating_ai_begruendung.loc[rating_ai_begruendung.index[0], begruendung_col]

            # Determine color based on mode
            if color_by_rating:
                # Color by reference rating type
                ref_rating_int = int(ref_rating) if not pd.isna(ref_rating) else 0
                color = rating_colors.get(ref_rating_int, '#95a5a6')
                color_category = rating_labels.get(ref_rating_int, f'Unknown ({ref_rating_int})')
            else:
                # Original behavior: color by LLM correct/wrong
                if ai_error_rate < human_error_rate or ai_error_rate <= 0.5:
                    color = '#2ca02c'  # green
                    color_category = 'LLM correct'
                else:
                    color = '#d62728'  # red
                    color_category = 'LLM wrong'

            # Who better (kept for backward compatibility)
            if ai_error_rate < human_error_rate or ai_error_rate <= 0.5:
                who_better = 'LLM correct'
            else:
                who_better = 'LLM wrong'

            begruendung_text_wrapped = _wrap_hover(begruendung_text, width=60)

            count_cols = [c for c in human_item_summary.columns if c.startswith('count_')]
            if count_cols:
                count_data = human_item_summary.loc[item_red, count_cols]
                max_count_col = count_data.idxmax()
                most_common_rating = int(max_count_col.replace('count_', ''))
            else:
                most_common_rating = np.nan

            rng = np.random.default_rng(hash(item_base) % 2**32)
            human_err_plot = human_error_rate

            if ai_error_rate <= 1e-9:
                jitter_y = rng.uniform(0, jitter_amount)
                ai_err_plot = min(100, ai_error_rate + jitter_y)
            elif ai_error_rate >= 100 - 1e-9:
                jitter_y = rng.uniform(0, jitter_amount)
                ai_err_plot = max(0, ai_error_rate - jitter_y)
            else:
                jitter_y = rng.uniform(-jitter_amount, jitter_amount)
                ai_err_plot = np.clip(ai_error_rate + jitter_y, 0, 100)

            plot_data.append({
                'item': item_base,
                'most_common_human_rating': most_common_rating,
                'human_error_rate': human_err_plot,
                'ai_error_rate': ai_err_plot,
                'human_error_rate_original': human_error_rate,
                'ai_error_rate_original': ai_error_rate,
                'ref_rating': ref_rating,
                'ai_prediction': ai_prediction,
                'color': color,
                'color_category': color_category,
                'who_better': who_better,
                'human_n_errors': int(human_item_summary.loc[item_red, 'n_errors']),
                'human_n_raters': int(human_item_summary.loc[item_red, 'n_raters']),
                'ai_n_errors': int(ai_item_summary.loc[item_red, 'n_errors']),
                'ai_n_raters': int(ai_item_summary.loc[item_red, 'n_raters']),
                'begruendung': begruendung_text_wrapped
            })

        df_plot = pd.DataFrame(plot_data)
        df_plot['video_id'] = vid
        df_plot['video_name'] = video_names.get(vid, f'Video {vid}')
        plot_data_dict[vid] = df_plot.copy()

        # Create figure
        fig = go.Figure()

        # Add threshold lines if enabled
        if show_threshold_lines:
            # Low error threshold (20%)
            fig.add_vline(
                x=low_error_threshold,
                line=dict(color='gray', width=2, dash='dash'),
                annotation_text=f'Low ({low_error_threshold}%)',
                annotation_position='top',
                annotation=dict(font_size=10, font_color='gray')
            )
            # High error threshold (80%)
            fig.add_vline(
                x=high_error_threshold,
                line=dict(color='gray', width=2, dash='dash'),
                annotation_text=f'High ({high_error_threshold}%)',
                annotation_position='top',
                annotation=dict(font_size=10, font_color='gray')
            )
            # Horizontal line at 50% (LLM correct/wrong boundary)
            fig.add_hline(
                y=50,
                line=dict(color='gray', width=1, dash='dot'),
            )

        # Determine grouping column based on color mode
        group_col = 'color_category' if color_by_rating else 'who_better'
        
        # Get unique categories and their colors
        if color_by_rating:
            categories = [rating_labels[k] for k in [0, 1, -99] if rating_labels[k] in df_plot['color_category'].values]
            category_colors = {rating_labels[k]: rating_colors[k] for k in [0, 1, -99]}
        else:
            categories = ['LLM correct', 'LLM wrong']
            category_colors = {'LLM correct': '#2ca02c', 'LLM wrong': '#d62728'}

        # Add scatter points for each category
        for idx, category in enumerate(categories):
            df_subset = df_plot[df_plot[group_col] == category]
            if df_subset.empty:
                continue

            # Hover text
            hover_text = []
            for _, row in df_subset.iterrows():
                text = (
                    f"<b>{row['item']}</b><br>"
                    f"<br>"
                    f"<b>Error rates:</b><br>"
                    f"Humans: {row['human_error_rate_original']:.1f}% ({row['human_n_errors']}/{row['human_n_raters']})<br>"
                    f"Most common human rating: {row['most_common_human_rating']}<br>"
                    f"LLM: {row['ai_error_rate_original']:.1f}% ({row['ai_n_errors']}/{row['ai_n_raters']})<br>"
                    f"<br>"
                    f"<b>Ratings:</b><br>"
                    f"Reference: {rating_labels.get(int(row['ref_rating']), row['ref_rating'])}<br>"
                    f"LLM prediction: {rating_labels.get(int(row['ai_prediction']), row['ai_prediction'])}<br>"
                    f"<br>"
                    f"<b>LLM justification:</b><br>"
                    f"{row['begruendung']}"
                )
                hover_text.append(text)

            fig.add_trace(go.Scatter(
                x=df_subset['human_error_rate'],
                y=df_subset['ai_error_rate'],
                mode='markers',
                name=category,
                marker=dict(
                    size=12,
                    color=category_colors.get(category, '#95a5a6'),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendrank=idx + 1
            ))

        # Layout
        vid_name = video_names[vid]
        legend_title = "Reference Rating" if color_by_rating else "LLM Performance"
        
        fig.update_layout(
            title=dict(
                text=f"{base_title}<br>Video {vid} - {vid_name}",
                font=dict(size=16, color='#333333')
            ),
            xaxis=dict(
                title="Human Error Rate (%)",
                range=[-5, 105],
                tickformat='.0f',
                ticksuffix='%',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title="LLM Correctness (Binary)",
                range=[-5, 105],
                tickvals=[0, 100],
                ticktext=["LLM correct", "LLM wrong"],
                tickformat='.0f',
                ticksuffix='%',
                gridcolor='lightgray',
                showgrid=True
            ),
            hovermode='closest',
            template='plotly_white',
            width=1200,
            height=700,
            legend=dict(
                title=dict(text=legend_title, font=dict(size=12)),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#333',
                borderwidth=1,
                x=1.00,
                y=0.50,
                xanchor='right',
                yanchor='middle'
            ),
            plot_bgcolor='#FAFAFA'
        )

        # Add quadrant annotations if threshold lines are shown
        if show_threshold_lines:
            # Get existing annotations (from add_vline, add_hline)
            existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
            
            # Add quadrant labels
            quadrant_annotations = [
                dict(x=0.08, y=0.92, xref='paper', yref='paper',
                     text='<b>LLM-specific<br>difficulty</b>', showarrow=False,
                     font=dict(size=11, color='#555555'), 
                     bgcolor='rgba(255,255,255,1.0)',
                     borderpad=4),
                dict(x=0.92, y=0.92, xref='paper', yref='paper',
                     text='<b>Shared<br>difficulty</b>', showarrow=False,
                     font=dict(size=11, color='#555555'),
                     bgcolor='rgba(255,255,255,1.0)',
                     borderpad=4),
                dict(x=0.08, y=0.08, xref='paper', yref='paper',
                     text='<b>Low<br>difficulty</b>', showarrow=False,
                     font=dict(size=11, color='#555555'),
                     bgcolor='rgba(255,255,255,1.0)',
                     borderpad=4),
                dict(x=0.92, y=0.08, xref='paper', yref='paper',
                     text='<b>Clinician-specific<br>difficulty</b>', showarrow=False,
                     font=dict(size=11, color='#555555'),
                     bgcolor='rgba(255,255,255,1.0)',
                     borderpad=4),
            ]
            
            # Combine existing and new annotations
            all_annotations = existing_annotations + quadrant_annotations
            fig.update_layout(annotations=all_annotations)

        # Save and show
        path_html_save = os.path.join(path_html, f"{prefix}_error_rate_scatter_video_{vid}.html")
        fig.write_html(path_html_save)
        fig.show()

        print(f"\n{'='*60}")
        print(f"Video {vid} - {vid_name}")
        print(f"{'='*60}")
        print(f"Total items: {len(df_plot)}")
        print(f"LLM correct: {len(df_plot[df_plot['who_better'] == 'LLM correct'])}")
        print(f"LLM wrong: {len(df_plot[df_plot['who_better'] == 'LLM wrong'])}")
        
        if color_by_rating:
            print(f"\nBy reference rating type:")
            for rating_val, label in rating_labels.items():
                count = len(df_plot[df_plot['ref_rating'] == rating_val])
                print(f"  {label}: {count}")

        figures.append(fig)

    return figures, plot_data_dict

def plot_difficulty_by_rating(
    difficulty_df: pd.DataFrame,
    video_id: int = None,
    video_name: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot reference rating distribution by difficulty type.
    
    Parameters:
    -----------
    difficulty_df : pd.DataFrame
        DataFrame with 'difficulty_type', 'ref_rating', and optionally 'video_id' columns
    video_id : int, optional
        If provided, filter for specific video. If None, use all videos.
    video_name : str, optional
        Name for the video (used in title). If None, uses video_id.
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.
        
    Returns:
    --------
    plt.Figure or None
    """
    # Define rating labels
    rating_labels = {0: 'Absent', 1: 'Present', -99: 'Not assessable'}
    
    # Filter for specific video if provided
    df = difficulty_df.copy()
    if video_id is not None and 'video_id' in df.columns:
        df = df[df['video_id'] == video_id]
    
    # Filter for the difficulty types we care about
    difficulty_types_of_interest = ['clinician_specific_difficulty', 'llm_specific_difficulty', 'shared_difficulty']
    filtered_df = df[df['difficulty_type'].isin(difficulty_types_of_interest)]
    
    if filtered_df.empty:
        print(f"No difficulty data found for video_id={video_id}")
        return None
    
    # Count by difficulty type and reference rating
    difficulty_counts = filtered_df.groupby(['difficulty_type', 'ref_rating']).size().unstack(fill_value=0)
    
    # Rename columns (ref_rating values) with labels
    difficulty_counts.columns = [rating_labels.get(c, str(c)) for c in difficulty_counts.columns]
    
    # Rename index (difficulty types) for cleaner labels
    difficulty_counts.index = difficulty_counts.index.map({
        'clinician_specific_difficulty': 'Clinician-Specific\nDifficulty',
        'llm_specific_difficulty': 'LLM-Specific\nDifficulty',
        'shared_difficulty': 'Shared\nDifficulty'
    })
    
    # Ensure all rating columns exist
    for col in ['Absent', 'Present', 'Not assessable']:
        if col not in difficulty_counts.columns:
            difficulty_counts[col] = 0
    
    # Reorder columns
    difficulty_counts = difficulty_counts[['Absent', 'Present', 'Not assessable']]
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    x = np.arange(len(difficulty_counts.index))
    width = 0.25
    multiplier = 0
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    
    for idx, (attribute, color) in enumerate(zip(difficulty_counts.columns, colors)):
        offset = width * multiplier
        bars = ax.bar(x + offset, difficulty_counts[attribute], width, label=attribute, color=color, alpha=0.8)
        # Add value labels on bars
        ax.bar_label(bars, padding=3, fontsize=10, fontweight='bold')
        multiplier += 1
    
    ax.set_xlabel('Difficulty Type', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    
    # Set title
    title_suffix = f"Video {video_id}" if video_id else "All Videos"
    if video_name:
        title_suffix = video_name
    ax.set_title(f'Reference Rating Distribution by Difficulty Type\n({title_suffix})', fontsize=14)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulty_counts.index, fontsize=11)
    ax.legend(title='Reference Rating', loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, difficulty_counts

def rater_level_ai_majority_vs_humans(
    master_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    ai_site_key: str = "best_ai_site",
    human_site_keys: list[str] = ["clinic_3","clinic_1","clinic_2"],
    vid_col: str = "video_id",
    ref_vid_col: str = "ID_Video",
    site_col: str = "site",
    id_col: str = "id_code_v2",
    psy_cols_regex: str = r'^p\d+_.*_final$',
    exclude_value: int = 10000,
    videos_to_use: list[int] | None = None,
    ai_rater_id: str = "AI_majority"
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Berechnet Accuracy pro Rater, wobei die KI als EIN Majority-Vote-Rater
    über alle KI-Zeilen pro Video/Item aggregiert wird.

    Rückgabe:
    - rater_df: DataFrame mit einer Zeile pro Rater:
        [rater_id, group, n_items, n_correct, accuracy]
    - summary: dict mit Gruppenstatistiken und Testresultaten
    - per_video_accuracy: DataFrame mit accuracy pro Person pro Video
        [rater_id, video_id, n_items, n_correct, accuracy]
    """
    psy_cols = [c for c in master_df.columns if re.match(psy_cols_regex, c)]

    ref_long = reference_df.melt(
        id_vars=[ref_vid_col],
        value_vars=psy_cols,
        var_name="item",
        value_name="ref_rating"
    ).rename(columns={ref_vid_col: vid_col})

    # Two accumulators: overall and per-video
    rater_stats: dict[str, dict] = {}
    per_video_stats: list[dict] = []  # NEW: track per video

    rater_stats[ai_rater_id] = {
        "group": "ai",
        "n_items": 0,
        "n_correct": 0,
    }

    for vid in sorted(master_df[vid_col].unique()):
        if videos_to_use is not None and vid not in videos_to_use:
            continue

        ref_rows = ref_long[ref_long[vid_col] == vid]
        if ref_rows.empty:
            continue

        df_vid = master_df[master_df[vid_col] == vid]
        if df_vid.empty:
            continue

        df_ai = df_vid[df_vid[site_col] == ai_site_key]
        if df_ai.empty:
            continue

        ref_series = ref_rows.set_index("item")["ref_rating"]

        usable_items = [
            itm for itm in psy_cols
            if itm in ref_series.index
            and pd.notna(ref_series.loc[itm])
            and ref_series.loc[itm] != exclude_value
        ]
        if not usable_items:
            continue

        ai_mode_row = df_ai[psy_cols].mode(axis=0)
        if ai_mode_row.empty:
            continue
        ai_vec = ai_mode_row.iloc[0]

        # AI per-video accuracy
        ai_video_items = 0
        ai_video_correct = 0
        for itm in usable_items:
            ref_val = ref_series.loc[itm]
            ai_val = ai_vec.get(itm, np.nan)

            if pd.isna(ai_val):
                continue
            if ai_val == exclude_value or ref_val == exclude_value:
                continue

            rater_stats[ai_rater_id]["n_items"] += 1
            ai_video_items += 1
            if ai_val == ref_val:
                rater_stats[ai_rater_id]["n_correct"] += 1
                ai_video_correct += 1

        # Store AI per-video result
        if ai_video_items > 0:
            per_video_stats.append({
                "rater_id": ai_rater_id,
                "video_id": vid,
                "group": "ai",
                "n_items": ai_video_items,
                "n_correct": ai_video_correct,
                "accuracy": ai_video_correct / ai_video_items
            })

        # Human raters
        df_hum = df_vid[df_vid[site_col].isin(human_site_keys)]
        if df_hum.empty:
            continue

        for _, row in df_hum.iterrows():
            r_id = row[id_col]
            group = "human"

            if r_id not in rater_stats:
                rater_stats[r_id] = {
                    "group": group,
                    "n_items": 0,
                    "n_correct": 0
                }
            else:
                if rater_stats[r_id]["group"] != group:
                    raise ValueError(
                        f"Rater {r_id} hat unterschiedliche Gruppenlabels"
                    )

            # Per-video counters for this human
            human_video_items = 0
            human_video_correct = 0

            for itm in usable_items:
                ref_val = ref_series.loc[itm]
                val = row.get(itm, np.nan)

                if pd.isna(val):
                    continue
                if val == exclude_value or ref_val == exclude_value:
                    continue

                rater_stats[r_id]["n_items"] += 1
                human_video_items += 1
                if val == ref_val:
                    rater_stats[r_id]["n_correct"] += 1
                    human_video_correct += 1

            # Store human per-video result
            if human_video_items > 0:
                per_video_stats.append({
                    "rater_id": r_id,
                    "video_id": vid,
                    "group": group,
                    "n_items": human_video_items,
                    "n_correct": human_video_correct,
                    "accuracy": human_video_correct / human_video_items
                })

    # Overall rater DataFrame
    rater_rows = []
    for r_id, stats in rater_stats.items():
        n_items = stats["n_items"]
        if n_items == 0:
            continue
        n_correct = stats["n_correct"]
        acc = n_correct / n_items
        rater_rows.append({
            "rater_id": r_id,
            "group": stats["group"],
            "n_items": n_items,
            "n_correct": n_correct,
            "accuracy": acc,
        })

    rater_df = pd.DataFrame(rater_rows)
    per_video_df = pd.DataFrame(per_video_stats)

    # Summary statistics (unchanged)
    summary: dict = {}
    if not rater_df.empty:
        summary["n_raters_total"] = int(len(rater_df))

        group_stats = (
            rater_df
            .groupby("group")["accuracy"]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n_raters"})
        )
        summary["group_stats"] = group_stats

        if set(rater_df["group"]) >= {"ai", "human"}:
            acc_ai = rater_df.loc[rater_df["group"] == "ai", "accuracy"].values
            acc_h = rater_df.loc[rater_df["group"] == "human", "accuracy"].values

            summary["n_ai_raters"] = int(len(acc_ai))
            summary["n_human_raters"] = int(len(acc_h))

            mean_ai = float(acc_ai.mean())
            mean_h = float(acc_h.mean())
            summary["mean_acc_ai"] = mean_ai
            summary["mean_acc_human"] = mean_h
            summary["mean_diff_ai_minus_human"] = mean_ai - mean_h

            if len(acc_ai) > 1 and len(acc_h) > 1:
                t_res = ttest_ind(acc_ai, acc_h, equal_var=False)
                summary["t_stat"] = float(t_res.statistic)
                summary["t_p"] = float(t_res.pvalue)
            else:
                summary["t_stat"] = np.nan
                summary["t_p"] = np.nan

            try:
                u_res = mannwhitneyu(acc_ai, acc_h, alternative="two-sided")
                summary["mw_u"] = float(u_res.statistic)
                summary["mw_p"] = float(u_res.pvalue)
            except ValueError:
                summary["mw_u"] = np.nan
                summary["mw_p"] = np.nan

            if len(acc_ai) > 1 and len(acc_h) > 1:
                var_ai = acc_ai.var(ddof=1)
                var_h = acc_h.var(ddof=1)
                pooled_sd = np.sqrt(
                    ((len(acc_ai) - 1) * var_ai + (len(acc_h) - 1) * var_h)
                    / (len(acc_ai) + len(acc_h) - 2)
                )
                cohens_d = (mean_ai - mean_h) / pooled_sd if pooled_sd > 0 else np.nan
            else:
                cohens_d = np.nan

            summary["cohens_d"] = float(cohens_d) if not np.isnan(cohens_d) else np.nan

    return rater_df, summary, per_video_df

def ai_vs_human_rater_winrate_from_rater_df(
    rater_df: pd.DataFrame,
    ai_group_label: str = "ai",
    human_group_label: str = "human",
    use_ai_mean: bool = False,
) -> dict:
    """
    Vergleicht KI mit allen menschlichen Ratern auf Rater-Level.
    Nutzt die Accuracy pro Rater und zählt, wie viele Rater von der KI
    'geschlagen' werden.

    Erwartete Struktur von rater_df:
      - Spalten: ['rater_id', 'group', 'n_items', 'n_correct', 'accuracy']
      - 'group' ist z.B. 'ai' oder 'human'

    use_ai_mean:
      - False: nimmt die erste KI-Zeile (z.B. Majority-Rater) als Referenz
      - True: Mittelwert aller KI-Rater (falls mehrere vorhanden sind)
    """
    ai_accs = rater_df.loc[rater_df["group"] == ai_group_label, "accuracy"].values
    if len(ai_accs) == 0:
        raise ValueError("Keine KI-Rater in rater_df gefunden (group == 'ai').")

    if use_ai_mean:
        ai_acc = float(ai_accs.mean())
    else:
        ai_acc = float(ai_accs[0])

    human_accs = rater_df.loc[rater_df["group"] == human_group_label, "accuracy"].values
    n_humans = len(human_accs)
    if n_humans == 0:
        raise ValueError("Keine menschlichen Rater in rater_df gefunden (group == 'human').")

    ai_better = int((ai_acc > human_accs).sum())
    human_better = int((ai_acc < human_accs).sum())
    ties = int((ai_acc == human_accs).sum())

    n_disc = ai_better + human_better

    if n_disc > 0:
        p_binom = binomtest(ai_better, n_disc, 0.5, alternative="two-sided").pvalue
    else:
        p_binom = float("nan")

    result = {
        "ai_accuracy_reference": ai_acc,
        "n_humans": int(n_humans),
        "ai_better": ai_better,
        "human_better": human_better,
        "ties": ties,
        "n_discordant": int(n_disc),
        "ai_win_fraction_over_raters": ai_better / n_disc if n_disc > 0 else float("nan"),
        "binom_p_over_raters": p_binom,
    }
    return result

def ai_majority_vs_human_rater_winrate_from_master(
    master_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    ai_site_key: str = "best_ai_site",
    human_site_keys: list[str] = ["clinic_3","clinic_1","clinic_2"],
    vid_col: str = "video_id",
    ref_vid_col: str = "ID_Video",
    site_col: str = "site",
    id_col: str = "id_code_v2",
    psy_cols_regex: str = r'^p\d+_.*_final$',
    exclude_value: int = 10000,
    videos_to_use: list[int] | None = None,
    ai_rater_id: str = "AI_majority",
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    Komplettpipeline für Majority-Vote-KI.

    Rückgabe:
    - rater_df: overall accuracy per rater
    - summary: group statistics
    - winrate_result: AI vs humans comparison
    - per_video_accuracy: accuracy per person per video
    """
    rater_df, summary, per_video_df = rater_level_ai_majority_vs_humans(
        master_df=master_df,
        reference_df=reference_df,
        ai_site_key=ai_site_key,
        human_site_keys=human_site_keys,
        vid_col=vid_col,
        ref_vid_col=ref_vid_col,
        site_col=site_col,
        id_col=id_col,
        psy_cols_regex=psy_cols_regex,
        exclude_value=exclude_value,
        videos_to_use=videos_to_use,
        ai_rater_id=ai_rater_id,
    )

    winrate_result = ai_vs_human_rater_winrate_from_rater_df(
        rater_df=rater_df,
        ai_group_label="ai",
        human_group_label="human",
        use_ai_mean=False,
    )

    return rater_df, summary, winrate_result, per_video_df

def get_ai_majority_vote_with_justifications(
    master_df: pd.DataFrame,
    ai_site_key: str,
    psy_cols_regex: str = r'^p\d+_.*_final$',
    vid_col: str = "video_id",
    site_col: str = "site",
    id_col: str = "ids"
) -> pd.DataFrame:
    """
    Calculate majority vote for AI ratings per video and item,
    including the justification from the first row that matches the majority vote.
    
    Returns a DataFrame with one row per video containing the majority vote
    for each psychology item plus corresponding justifications.
    """
    # Extract psychology columns
    psy_cols = [c for c in master_df.columns if re.match(psy_cols_regex, c)]
    
    # Get corresponding justification columns
    begr_cols = [c.split('_')[0] + '_begründung' for c in psy_cols]
    # Filter to only existing begründung columns
    begr_cols = [c for c in begr_cols if c in master_df.columns]
    
    # Filter to AI site only
    ai_df = master_df[master_df[site_col] == ai_site_key].copy()
    
    if ai_df.empty:
        raise ValueError(f"No data found for AI site: {ai_site_key}")
    
    # Calculate majority vote per video
    majority_rows = []
    for vid in ai_df[vid_col].unique():
        vid_data = ai_df[ai_df[vid_col] == vid]
        
        # Get mode for each psychology column
        majority_vote = vid_data[psy_cols].mode(axis=0).iloc[0]
        
        # Create result row
        result_row = {
            vid_col: vid,
            id_col: 0,  # Special ID for majority vote
            site_col: f"{ai_site_key}_majority"
        }
        result_row.update(majority_vote.to_dict())
        
        # For each psychology item, find matching justification
        for psy_col in psy_cols:
            # Get the majority rating for this item
            maj_rating = majority_vote[psy_col]
            
            # Find corresponding begründung column
            begr_col = psy_col.split('_')[0] + '_begründung'
            
            if begr_col in begr_cols:
                # Find first row where rating matches majority
                matching_rows = vid_data[vid_data[psy_col] == maj_rating]
                
                if not matching_rows.empty:
                    # Take justification from first matching row
                    first_match_justification = matching_rows[begr_col].iloc[0]
                    result_row[begr_col + "_ai"] = first_match_justification
                else:
                    # No matching row found (shouldn't happen, but handle it)
                    result_row[begr_col] = vid_data[begr_col].iloc[0]
        
        majority_rows.append(result_row)
    
    return pd.DataFrame(majority_rows)

def compute_paired_comparison_stats(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group1",
    group2_name: str = "Group2"
) -> dict:
    """
    Compute comprehensive statistics and effect sizes for paired comparisons.
    
    Parameters:
    -----------
    group1, group2 : array-like
        Paired samples to compare
    group1_name, group2_name : str
        Names for the groups (for labeling results)
    
    Returns:
    --------
    dict with statistics including:
        - mean_diff: Mean difference (group1 - group2)
        - t_test_p: Paired t-test p-value
        - wilcoxon_p: Wilcoxon signed-rank test p-value
        - cohens_d: Cohen's d effect size
        - r_effect_size: Correlation-based effect size
        - cles: Common Language Effect Size (probability group1 > group2)
        - cliffs_delta: Cliff's Delta (non-parametric effect size)
        - group1_better_count: Number of pairs where group1 > group2
    """
    from scipy.stats import ttest_rel, wilcoxon
    
    # Convert to arrays
    arr1 = np.array(group1)
    arr2 = np.array(group2)
    
    # Basic statistics
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    mean_diff = mean1 - mean2
    
    # Statistical tests
    t_stat, t_p = ttest_rel(arr1, arr2)
    w_stat, w_p = wilcoxon(arr1, arr2)
    
    # Effect sizes
    # 1. Cohen's d for paired samples
    differences = arr1 - arr2
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    
    # 2. Correlation-based effect size (for paired t-test)
    df = len(arr1) - 1
    r_effect = np.sqrt(t_stat**2 / (t_stat**2 + df))
    
    # 3. Common Language Effect Size (CLES)
    n_group1_better = np.sum(arr1 > arr2)
    n_total = len(arr1)
    cles = n_group1_better / n_total
    
    # 4. Cliff's Delta (non-parametric effect size)
    def cliffs_delta_paired(x, y):
        """Calculate Cliff's Delta for paired data"""
        n = len(x)
        gt = sum(1 for i in range(n) if x[i] > y[i])
        lt = sum(1 for i in range(n) if x[i] < y[i])
        return (gt - lt) / n
    
    cliff_d = cliffs_delta_paired(arr1, arr2)
    
    return {
        'comparison': f"{group1_name}_vs_{group2_name}",
        'n_pairs': int(n_total),
        'mean_group1': float(mean1),
        'mean_group2': float(mean2),
        'mean_diff': float(mean_diff),
        't_test_statistic': float(t_stat),
        't_test_p': float(t_p),
        'wilcoxon_statistic': float(w_stat),
        'wilcoxon_p': float(w_p),
        'group1_better_count': int(n_group1_better),
        'group2_better_count': int(np.sum(arr1 < arr2)),
        'ties_count': int(np.sum(arr1 == arr2)),
        # Effect sizes
        'cohens_d': float(cohens_d),
        'r_effect_size': float(r_effect),
        'cles': float(cles),
        'cliffs_delta': float(cliff_d)
    }

def compute_all_pairwise_comparisons(
    results: dict,
    summary: dict,
    comparison_pairs: list[tuple[str, str]] = None
) -> dict:
    """
    Compute all pairwise comparisons with comprehensive statistics.
    
    Parameters:
    -----------
    results : dict
        Dictionary with strategy names as keys and accuracy arrays as values
    summary : dict
        Dictionary with summary statistics for each strategy
    comparison_pairs : list of tuples, optional
        List of (strategy1, strategy2) pairs to compare.
        If None, compares all combinations.
    
    Returns:
    --------
    dict with comparison results
    """
    comparisons = {}
    
    # Default comparisons if not specified
    if comparison_pairs is None:
        strategies = list(results.keys())
        comparison_pairs = [
            ('ai_always', 'human_only'),
            ('ai_tiebreaker', 'human_only'),
            ('ai_always', 'ai_tiebreaker'),
            ('ai_always', 'ai_only'),
            
        ]
        # Filter to only include existing strategies
        comparison_pairs = [
            (s1, s2) for s1, s2 in comparison_pairs
            if s1 in strategies and s2 in strategies
        ]
    
    # Compute stats for each comparison
    for strategy1, strategy2 in comparison_pairs:
        if strategy1 not in results or strategy2 not in results:
            continue
            
        stats = compute_paired_comparison_stats(
            group1=results[strategy1],
            group2=results[strategy2],
            group1_name=strategy1,
            group2_name=strategy2
        )
        
        comp_name = f"{strategy1}_vs_{strategy2}"
        comparisons[comp_name] = stats
    
    return comparisons

def simulate_all_human_pairs_with_ai(
    master_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    ai_site_key: str,
    human_site_keys: list[str],
    psy_cols: list = [],
    vid_col: str = "video_id",
    ref_vid_col: str = "ID_Video",
    site_col: str = "site",
    id_col: str = "id_code_v2",
    exclude_value: int = 10000,
    videos_to_use: list[int] = None,
    haupttätigkeit_col: str = "haupttätigkeit_v2",
    psychiatrist_codes: list[int] = [2, 3, 5]  # Trained psychiatrists
) -> dict:
    """
    Test ALL possible human pairs that rated the same video (exhaustive, not random sampling).
    
    Simulate three voting strategies:
    1. AI-always: When humans disagree, always use AI vote
    2. Human-only: When humans disagree, randomly choose between the two humans
    3. Human-only-supervision: When humans disagree, consult a random trained psychiatrist
    4. AI-only: Always use AI (baseline)
    
    Returns accuracy statistics for each strategy.
    """
    
    # Get reference ratings
    ref_long = reference_df.melt(
        id_vars=[ref_vid_col],
        value_vars=psy_cols,
        var_name="item",
        value_name="ref_rating"
    ).rename(columns={ref_vid_col: vid_col})
    
    # Filter to selected videos
    if videos_to_use is not None:
        master_df = master_df[master_df[vid_col].isin(videos_to_use)]
        ref_long = ref_long[ref_long[vid_col].isin(videos_to_use)]
    
    disagreement_log = []
    results = {
        'ai_always': [],
        'human_only': [],
        'human_only_supervision': [],
        "ai_only": []
    }
    
    simulation_details = []
    
    # For each video
    for vid in sorted(master_df[vid_col].unique()):
        ref_rows = ref_long[ref_long[vid_col] == vid]
        if ref_rows.empty:
            continue

        df_vid = master_df[master_df[vid_col] == vid]
        
        # Get AI ratings
        df_ai = df_vid[df_vid[site_col] == ai_site_key]
        if df_ai.empty:
            continue
        ai_mode_row = df_ai[psy_cols].mode(axis=0).iloc[0]
        
        # Get all human raters for this video
        df_humans = df_vid[df_vid[site_col].isin(human_site_keys)]
        
        # Identify trained psychiatrists (for consultation only)
        psychiatrist_mask = df_humans[haupttätigkeit_col].isin(psychiatrist_codes)
        df_psychiatrists = df_humans[psychiatrist_mask].copy()
        psychiatrist_ids = df_psychiatrists[id_col].unique()
        
        # Get non-psychiatrist raters (for pairing)
        df_non_psychiatrists = df_humans[~psychiatrist_mask].copy()
        non_psychiatrist_ids = df_non_psychiatrists[id_col].unique()
        
        if len(non_psychiatrist_ids) < 2:
            print(f"Video {vid}: Need at least 2 non-psychiatrist raters. Skipping.")
            continue
        
        if len(psychiatrist_ids) == 0:
            print(f"Video {vid}: No trained psychiatrists available for consultation. Skipping.")
            continue
        
        ref_series = ref_rows.set_index("item")["ref_rating"]
        
        # Get valid items
        valid_items = [
            itm for itm in psy_cols
            if itm in ref_series.index
            and pd.notna(ref_series.loc[itm])
            and ref_series.loc[itm] != exclude_value
        ]
        
        if not valid_items:
            continue
        
        # Generate ALL possible pairs of NON-PSYCHIATRIST raters for this video
        all_pairs = list(combinations(non_psychiatrist_ids, 2))
        
        # Test each pair
        for pair_idx, (h1, h2) in enumerate(all_pairs):
            h1_ratings = df_non_psychiatrists[df_non_psychiatrists[id_col] == h1][psy_cols].iloc[0]
            h2_ratings = df_non_psychiatrists[df_non_psychiatrists[id_col] == h2][psy_cols].iloc[0]
            
            pair_valid_items = [
                itm for itm in valid_items
                if pd.notna(h1_ratings.get(itm)) 
                and pd.notna(h2_ratings.get(itm))
                and h1_ratings.get(itm) != exclude_value
                and h2_ratings.get(itm) != exclude_value
                and pd.notna(ai_mode_row.get(itm))
                and ai_mode_row.get(itm) != exclude_value
            ]
            
            if not pair_valid_items:
                continue
                
            sim_correct = {
                'ai_always': 0,
                'human_only': 0,
                'human_only_supervision': 0,
                'ai_only': 0
            }
            sim_items = 0
            sim_disagreements = 0
            sim_ai_used = 0
            sim_psychiatrist_consulted = 0
            
            for itm in pair_valid_items:
                ref_val = ref_series.loc[itm]
                h1_val = float(h1_ratings.get(itm, np.nan))
                h2_val = float(h2_ratings.get(itm, np.nan))
                ai_val = float(ai_mode_row.get(itm, np.nan))
                ref_val = float(ref_val)
                
                # Skip if any value is invalid
                if any(pd.isna(x) or x == exclude_value for x in [h1_val, h2_val, ai_val, ref_val]):
                    continue
                
                sim_items += 1
                sim_correct["ai_only"] += int(ai_val == ref_val)
                

                # Check if humans agree
                if h1_val == h2_val:
                    # All strategies use same value when humans agree
                    final_val = h1_val
                    sim_correct['ai_always'] += int(final_val == ref_val)
                    sim_correct['human_only'] += int(final_val == ref_val)
                    sim_correct['human_only_supervision'] += int(final_val == ref_val)

                else:
                    # Humans disagree
                    sim_disagreements += 1
                    
                    # Strategy 1: Always use AI
                    ai_is_correct = (ai_val == ref_val)
                    sim_correct['ai_always'] += int(ai_val == ref_val)
                    sim_ai_used += 1
                    
                    # Strategy 2: Random human choice
                    random_human_val = np.random.choice([h1_val, h2_val])
                    human_is_correct = (random_human_val == ref_val)
                    sim_correct['human_only'] += int(random_human_val == ref_val)
                    
                    # Strategy 3: Consult random trained psychiatrist
                    available_psychiatrists = []
                    for psych_id in psychiatrist_ids:
                        psych_rating = df_psychiatrists[df_psychiatrists[id_col] == psych_id][itm].iloc[0]
                        if pd.notna(psych_rating) and psych_rating != exclude_value:
                            available_psychiatrists.append((psych_id, float(psych_rating)))
                    
                    if available_psychiatrists:
                        # Randomly select one psychiatrist
                        _, psychiatrist_val = available_psychiatrists[
                            np.random.randint(len(available_psychiatrists))
                        ]
                        final_val = psychiatrist_val
                        sim_psychiatrist_consulted += 1
                    else:
                        # Fallback: no psychiatrist available, random choice between h1 and h2
                        final_val = np.random.choice([h1_val, h2_val])
                    
                    supervision_is_correct = (final_val == ref_val)
                    sim_correct['human_only_supervision'] += int(final_val == ref_val)

                    disagreement_log.append({
                        'video_id': vid,
                        'pair_id': pair_idx,
                        'item_id': itm,
                        'strategy_chosen': 'ai_always',
                        'final_value': ai_val,
                        'correct_reference': ref_val,
                        'is_correct': ai_is_correct
                    })
                    
                    # Log Human Random Strategy
                    disagreement_log.append({
                        'video_id': vid,
                        'pair_id': pair_idx,
                        'item_id': itm,
                        'strategy_chosen': 'human_only',
                        'final_value': random_human_val,
                        'correct_reference': ref_val,
                        'is_correct': human_is_correct
                    })
                    
                    # Log Supervision Strategy
                    disagreement_log.append({
                        'video_id': vid,
                        'pair_id': pair_idx,
                        'item_id': itm,
                        'strategy_chosen': 'human_only_supervision',
                        'final_value': final_val,
                        'correct_reference': ref_val,
                        'is_correct': supervision_is_correct
                    })
            if sim_items > 0:
                simulation_details.append({
                    'video_id': vid,
                    'pair_id': pair_idx,
                    'human1': h1,
                    'human2': h2,
                    'n_items': sim_items,
                    'n_disagreements': sim_disagreements,
                    'disagreement_rate': sim_disagreements / sim_items,
                    'ai_used': sim_ai_used,
                    'psychiatrist_consulted': sim_psychiatrist_consulted,
                    'accuracy_ai_always': sim_correct['ai_always'] / sim_items,
                    'accuracy_human_only': sim_correct['human_only'] / sim_items,
                    'accuracy_human_only_supervision': sim_correct['human_only_supervision'] / sim_items,
                    'accuracy_ai_only': sim_correct['ai_only'] / sim_items
                })
                
                for strategy in results.keys():
                    results[strategy].append(sim_correct[strategy] / sim_items)
    
    # Convert to arrays and compute statistics
    summary = {}
    for strategy, accuracies in results.items():
        arr = np.array(accuracies)
        summary[strategy] = {
            'mean_accuracy': float(arr.mean()),
            'std_accuracy': float(arr.std()),
            'median_accuracy': float(np.median(arr)),
            'n_pairs': len(arr)
        }
    comparison_pairs =[ ('ai_always', 'human_only'),
            ('human_only_supervision', 'human_only'),
            ('ai_always', 'human_only_supervision')]

    # Statistical comparisons
    comparisons = compute_all_pairwise_comparisons(results, summary, comparison_pairs)
    df_disagreements = pd.DataFrame(disagreement_log)
    return {
        'summary': summary,
        'comparisons': comparisons,
        'raw_results': results,
        'simulation_details': pd.DataFrame(simulation_details),
        'disagreement_data': df_disagreements
    }

def print_comparison_results(comparisons: dict, alpha: float = 0.05):
    """
    Print formatted comparison results with effect size interpretations.
    
    Parameters:
    -----------
    comparisons : dict
        Output from compute_all_pairwise_comparisons
    alpha : float
        Significance level for hypothesis tests
    """
    print("\n" + "="*80)
    print("PAIRWISE COMPARISON RESULTS WITH EFFECT SIZES")
    print("="*80)
    
    for comp_name, stats in comparisons.items():
        print(f"\n{comp_name.upper().replace('_', ' ')}")
        print("-" * 60)
        
        # Sample info
        print(f"Sample size: {stats['n_pairs']} paired observations")
        
        # Means and difference
        print(f"\nMean {stats['comparison'].split('_vs_')[0]}: {stats['mean_group1']:.4f}")
        print(f"Mean {stats['comparison'].split('_vs_')[1]}: {stats['mean_group2']:.4f}")
        print(f"Mean difference: {stats['mean_diff']:.4f}")
        
        # Counts
        group1_name = stats['comparison'].split('_vs_')[0]
        group2_name = stats['comparison'].split('_vs_')[1]
        print(f"\n{group1_name} better: {stats['group1_better_count']}/{stats['n_pairs']}")
        print(f"{group2_name} better: {stats['group2_better_count']}/{stats['n_pairs']}")
        print(f"Ties: {stats['ties_count']}/{stats['n_pairs']}")
        
        # Statistical tests
        print(f"\nStatistical Tests:")
        t_sig = "***" if stats['t_test_p'] < alpha else ""
        w_sig = "***" if stats['wilcoxon_p'] < alpha else ""
        print(f"  Paired t-test:  t = {stats['t_test_statistic']:.3f}, p = {stats['t_test_p']:.4g} {t_sig}")
        print(f"  Wilcoxon test:  W = {stats['wilcoxon_statistic']:.1f}, p = {stats['wilcoxon_p']:.4g} {w_sig}")
        
        # Effect sizes with interpretations
        print(f"\nEffect Sizes:")
        
        # Cohen's d
        d = abs(stats['cohens_d'])
        d_interp = "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
        print(f"  Cohen's d:        {stats['cohens_d']:+.3f} ({d_interp})")
        print(f"    Interpretation: Small=0.2, Medium=0.5, Large=0.8")
        
        # r effect size
        r = abs(stats['r_effect_size'])
        r_interp = "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"
        print(f"  r effect size:    {stats['r_effect_size']:+.3f} ({r_interp})")
        print(f"    Interpretation: Small=0.1, Medium=0.3, Large=0.5")
        
        # CLES
        print(f"  CLES (P(G1>G2)):  {stats['cles']:.3f}")
        print(f"    Interpretation: Probability that {group1_name} > {group2_name}")
        
        # Cliff's Delta
        cliff = abs(stats['cliffs_delta'])
        cliff_interp = "negligible" if cliff < 0.147 else "small" if cliff < 0.33 else "medium" if cliff < 0.474 else "large"
        print(f"  Cliff's Delta:    {stats['cliffs_delta']:+.3f} ({cliff_interp})")
        print(f"    Interpretation: Small=0.147, Medium=0.33, Large=0.474")


def run_simulation_per_video(
    master_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    ai_site_key: str,
    human_site_keys: list[str],
    psy_cols: list,
    vid_col: str = "video_id",
    ref_vid_col: str = "ID_Video",
    site_col: str = "site",
    id_col: str = "id_code_v2",
    exclude_value: int = 10000,
    haupttätigkeit_col: str = "haupttätigkeit_v2",
    psychiatrist_codes: list[int] = [2, 3, 5],
    videos_to_analyze : list[int] | None = None
) -> dict:
    """
    Run simulations separately for each video.
    Returns dict with video_id as key and simulation results as value.
    """
    video_results = {}
    if videos_to_analyze is None:
        videos_to_analyze = sorted(master_df[vid_col].unique())
    
    for vid in videos_to_analyze:
        print(f"\nProcessing Video {vid}...")
        
        # Run simulation for this specific video
        sim_result = simulate_all_human_pairs_with_ai(
            master_df=master_df,
            reference_df=reference_df,
            ai_site_key=ai_site_key,
            human_site_keys=human_site_keys,
            psy_cols=psy_cols,
            vid_col=vid_col,
            ref_vid_col=ref_vid_col,
            site_col=site_col,
            id_col=id_col,
            exclude_value=exclude_value,
            videos_to_use=[vid],  # Only this video
            haupttätigkeit_col=haupttätigkeit_col,
            psychiatrist_codes=psychiatrist_codes
        )
        
        video_results[vid] = sim_result
    
    return video_results


def plot_per_video_comparison(
    video_results: dict,
    human_per_video: pd.DataFrame,
    ai_per_video: pd.DataFrame,
    video_names: dict = {7: 'Mania', 8: 'Depression', 9: 'Schizophrenia'},
    figsize: tuple = (10, 6)
):
    """
    Create a plot for each video showing the three strategies with reference lines.
    """
    for vid, sim_result in video_results.items():
        vid_name = video_names.get(vid, f"Video {vid}")
        
        # Prepare data for plotting
        raw_results = sim_result['raw_results']
        melted_data = pd.DataFrame({
            'Strategy': ['human_only'] * len(raw_results['human_only']) + 
                       ['human_only_supervision'] * len(raw_results['human_only_supervision']) + 
                       ['ai_always'] * len(raw_results['ai_always']),
            'Accuracy': raw_results['human_only'] + 
                       raw_results['human_only_supervision'] + 
                       raw_results['ai_always']
        })
        
        # Get human and AI means for this video
        human_vid_accs = human_per_video[human_per_video['video_id'] == vid]['accuracy'].values
        ai_vid_accs = ai_per_video[ai_per_video['video_id'] == vid]['accuracy'].values
        
        if len(human_vid_accs) == 0 or len(ai_vid_accs) == 0:
            print(f"Warning: No data for video {vid}")
            continue
        
        human_mean = human_vid_accs.mean()
        ai_mean = ai_vid_accs.mean()
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Violin plot
        sns.violinplot(
            data=melted_data,
            x="Strategy",
            y="Accuracy",
            order=["human_only", "human_only_supervision", "ai_always"],
            ax=ax
        )
        
        # Mean markers
        means = melted_data.groupby("Strategy", as_index=False)["Accuracy"].mean()
        sns.scatterplot(
            data=means,
            x="Strategy",
            y="Accuracy",
            marker="D",
            s=100,
            color="green",
            zorder=10,
            ax=ax
        )
        
        # Reference lines
        ax.axhline(y=human_mean, color='darkblue', linestyle='--', 
                   label=f'Clinician Mean ({human_mean:.2f})', linewidth=2)
        ax.axhline(y=ai_mean, color='orange', linestyle='--', 
                   label=f'AI - Majority Mean ({ai_mean:.2f})', linewidth=2)
        
        # Styling
        #ax.set_title(f"Distribution of Accuracy Across Simulations\nVideo {vid} - {vid_name}", 
        #             fontsize=14, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xticklabels([
            'Clinicians Only',
            'Clinicians (Board-Certified Supervision)',
            'Clinicians (AI-Supervision)'
        ], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics for this video
        print(f"\n{'='*60}")
        print(f"VIDEO {vid} - {vid_name.upper()} STATISTICS")
        print(f"{'='*60}")
        print(f"Human mean accuracy: {human_mean:.4f}")
        print(f"AI majority accuracy: {ai_mean:.4f}")
        print(f"\nSimulation results:")
        for strategy, stats in sim_result['summary'].items():
            print(f"  {strategy}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        
        print(f"\nConsultation statistics:")
        details = sim_result['simulation_details']
        print(f"  Total pairs: {len(details)}")
        print(f"  Disagreement rate: {details['disagreement_rate'].mean():.2%}")
        print(f"  AI consultations: {details['ai_used'].sum():.0f}")
        print(f"  Psychiatrist consultations: {details['psychiatrist_consulted'].sum():.0f}")




def plot_human_vs_ai_violin(
    human_accs: np.ndarray,
    ai_accuracy: float,
    video_id: int | str = "All Videos",
    figsize: tuple = (6, 8),
    show_legend: bool = True,
    ax: plt.Axes = None,
    return_fig: bool = False
) -> plt.Axes | tuple:
    """
    Create a violin plot comparing human rater accuracies with AI performance.
    
    Parameters:
    -----------
    human_accs : np.ndarray
        Array of human rater accuracies
    ai_accuracy : float
        Single AI accuracy value
    video_id : int or str
        Video identifier for the title
    figsize : tuple
        Figure size (width, height)
    show_legend : bool
        Whether to show the legend
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure
    return_fig : bool
        If True, returns (fig, ax). If False, returns only ax
    
    Returns:
    --------
    plt.Axes or tuple
        If return_fig=True: (fig, ax)
        If return_fig=False: ax
    """
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create violin plot
    parts = ax.violinplot(
        [human_accs],
        positions=[0],
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )
    
    # Color the violin
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add scatter points for individual humans
    y_scatter = np.random.normal(0, 0.03, size=len(human_accs))
    ax.scatter(
        y_scatter, human_accs,
        alpha=0.6, s=80, color="#0f5586",
        edgecolor='white', linewidth=1,
        label='Human raters'
    )
    
    # Add AI diamond marker
    ax.scatter(
        [0], [ai_accuracy],
        marker='D', s=100, color='#d62728',
        label=f'LLM ({ai_accuracy:.3f})', zorder=5,
        edgecolor='black', linewidth=1
    )
    
    # Add reference lines
    ax.axhline(
        human_accs.mean(), color='darkblue', linestyle='--',
        linewidth=2, alpha=0.7, label=f'Human mean ({human_accs.mean():.3f})'
    )
    ax.axhline(
        np.median(human_accs), color='orange', linestyle='--',
        linewidth=2, alpha=0.7, label=f'Human median ({np.median(human_accs):.3f})'
    )
    
    # Styling
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(human_accs.min() - 0.05, human_accs.max() + 0.05)
    ax.set_xticks([])
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax.set_title(
        f'{video_id} — Distribution of Human Raters with LLM Performance',
        fontweight='bold', fontsize=12
    )
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    
    # Add legend with smaller marker sizes
    if show_legend:
        legend = ax.legend(loc='lower right', fontsize=9,
                          handlelength=1.5, scatterpoints=1)
        
        # Scale down marker sizes in the legend
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_markersize'):
                handle.set_markersize(5)
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([25])
    
    if return_fig:
        return fig, ax
    return ax




def create_publication_error_scatter_with_extreme_cases(
    master_df: pd.DataFrame,
    reference: pd.DataFrame,
    best_ai: str,
    psy_cols: list,
    groups_dict_reduced: dict = None,
    video_ids: list = [7, 8, 9],
    jitter_amount: float = 30,
    base_title: str = "Error Rate Comparison: LLM vs. Humans",
    save: bool = False,
):
    """
    Create publication-ready scatter plots with extreme disagreement analysis.
    
    For each video:
    - Row 1: Scatter plot (Human error on x-axis, LLM error on y-axis)
             with 4 circled extreme cases
    - Row 2: Four bar plots showing human rating distributions for extreme cases
             with markers for AI prediction (diamond) and reference (star)
    
    Extreme cases:
    - 2 items where AI error = 0% and human error is maximal
    - 2 items where AI error = 100% and human error is minimal
    """
    if groups_dict_reduced is None:
        groups_dict_reduced = {
            "all_items": [0, 1, -99],
        }
    # English video names
    video_names = {7: 'Mania', 8: 'Depression', 9: 'Schizophrenia'}
    
    # Rating label mapping
    rating_labels = {0: 'Absent', 1: 'Present', -99: 'Not assessable'}
    
    figures = []
    
    for vid in video_ids:
        # Filter to single video
        master_vid = master_df[master_df['video_id'] == vid].copy()
        ref_vid = reference[reference['ID_Video'] == vid].copy()
        
        # Human and LLM ratings for this video
        rating_humans_copy = master_vid[master_vid['site'].isin(['clinic_3', 'clinic_1', 'clinic_2'])].copy()
        rating_ai_all = master_vid[master_vid['site'] == best_ai][psy_cols].copy()
        rating_ref = ref_vid[psy_cols].copy()
        
        # Get LLM justifications
        ai_begruendung_cols = [col.split('_')[0] + "_begründung" for col in psy_cols]
        rating_ai_begruendung = master_vid[master_vid['site'] == best_ai][ai_begruendung_cols].copy()
        
        # Reduced versions (3 categories)
        rating_ai_mode = rating_ai_all.mode(axis=0).iloc[0].to_frame().T
        rating_ai_mode['ids'] = 0
        
        all_dfs = [rating_humans_copy, rating_ai_mode, rating_ref, rating_ai_all]
        reduced_dfs = reduce_data_to_3_categories(psy_cols, all_dfs)
        
        rating_humans_copy_red = reduced_dfs[0]
        rating_ai_red = reduced_dfs[1]
        rating_ref_red = reduced_dfs[2]
        rating_ai_all_red = reduced_dfs[3]
        
        psy_cols_red = [c + "_reduced" for c in psy_cols]
        
        # Compute accuracies
        human_results = compute_multiple_accuracies_for_filtered_rating_ranges(
            rating_ref=rating_ref_red[psy_cols_red],
            rating_df=rating_humans_copy_red[psy_cols_red + ['ids']],
            groups_dict=groups_dict_reduced,
            id_col='ids'
        )
        ai_results = compute_multiple_accuracies_for_filtered_rating_ranges(
            rating_ref=rating_ref_red[psy_cols_red],
            rating_df=rating_ai_red[psy_cols_red + ['ids']],
            groups_dict=groups_dict_reduced,
            id_col='ids'
        )
        
        # Item summaries
        human_item_summary = human_results['all_items'][4]
        ai_item_summary = ai_results['all_items'][4]
        
        # Prepare data for plotting
        plot_data = []
        for item_red in psy_cols_red:
            item_base = item_red.replace('_reduced', '')
            begruendung_col = item_base.split('_')[0] + "_begründung"
            maj_rating = rating_ai_red[item_red].iloc[0]
            matching_rows = rating_ai_all_red[rating_ai_all_red[item_red] == maj_rating]
            if not matching_rows.empty:
                # Get the index of the first matching row
                first_match_idx = matching_rows.index[0]
                ai_justification = rating_ai_begruendung.loc[first_match_idx, begruendung_col]
            else:
                print(f"had to use fallback for item {item_base}")
                # Fallback to first row if no match (shouldn't happen)
                ai_justification = rating_ai_begruendung.loc[rating_ai_begruendung.index[0], begruendung_col]
            # Error rates
            human_error_rate = human_item_summary.loc[item_red, 'error_rate'] * 100
            ai_error_rate = ai_item_summary.loc[item_red, 'error_rate'] * 100
            
            # Reference and prediction (reduced)
            ref_rating = rating_ref_red[item_red].iloc[0]
            ai_prediction = rating_ai_red[item_red].iloc[0]
            
            # Category and color
            if ai_error_rate < human_error_rate or ai_error_rate <= 0.5:
                color = '#2ca02c'  # green
                who_better = 'LLM correct'
            elif ai_error_rate > human_error_rate or ai_error_rate >= 0.5:
                color = '#d62728'  # red
                who_better = 'LLM wrong'
            
            # Get count columns for human ratings
            count_cols = [c for c in human_item_summary.columns if c.startswith('count_')]
            count_data = {}
            if count_cols:
                for cc in count_cols:
                    rating_val = int(cc.replace('count_', ''))
                    count_data[rating_val] = int(human_item_summary.loc[item_red, cc])
            
            # Jitter for plotting (same logic as original)
            rng = np.random.default_rng(hash(item_base) % 2**32)
            human_err_plot = human_error_rate
            
            if ai_error_rate <= 1e-9:
                jitter_y = rng.uniform(0, jitter_amount)
                ai_err_plot = min(100, ai_error_rate + jitter_y)
            elif ai_error_rate >= 100 - 1e-9:
                jitter_y = rng.uniform(0, jitter_amount)
                ai_err_plot = max(0, ai_error_rate - jitter_y)
            else:
                jitter_y = rng.uniform(-jitter_amount, jitter_amount)
                ai_err_plot = np.clip(ai_error_rate + jitter_y, 0, 100)
            
            plot_data.append({
                'item': item_base,
                'human_error_rate': human_err_plot,
                'ai_error_rate': ai_err_plot,
                'human_error_rate_original': human_error_rate,
                'ai_error_rate_original': ai_error_rate,
                'ref_rating': ref_rating,
                'ai_prediction': ai_prediction,
                'color': color,
                'who_better': who_better,
                'human_n_errors': int(human_item_summary.loc[item_red, 'n_errors']),
                'human_n_raters': int(human_item_summary.loc[item_red, 'n_raters']),
                'count_data': count_data,
                'ai_justification': ai_justification 
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # ============ SELECT EXTREME CASES ============
        tolerance = 1e-9
        
        # Type A: AI perfect (error ≈ 0), humans maximally wrong
        df_ai_perfect = df_plot[df_plot['ai_error_rate_original'] <= tolerance].copy()
        if len(df_ai_perfect) >= 2:
            df_ai_perfect = df_ai_perfect.nlargest(2, 'human_error_rate_original')
        extreme_A = df_ai_perfect.to_dict('records')
        
        # Type B: AI completely wrong (error ≈ 100), humans minimally wrong
        df_ai_wrong = df_plot[df_plot['ai_error_rate_original'] >= (100 - tolerance)].copy()
        if len(df_ai_wrong) >= 2:
            df_ai_wrong = df_ai_wrong.nsmallest(2, 'human_error_rate_original')
        extreme_B = df_ai_wrong.to_dict('records')
        
        extreme_cases = extreme_A + extreme_B
        
        # ============ CREATE FIGURE ============
        fig = plt.figure(figsize=(17, 10))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.35, wspace=0.3,
                             left=0.08, right=0.88, top=0.92, bottom=0.08)
        
        # Row 1: Scatter plot (spans all columns)
        ax_scatter = fig.add_subplot(gs[0, :])
        
        # Plot all points
        for who_better in ['LLM correct', 'LLM wrong']:
            df_subset = df_plot[df_plot['who_better'] == who_better]
            if df_subset.empty:
                continue
            ax_scatter.scatter(
                df_subset['human_error_rate'],
                df_subset['ai_error_rate'],
                s=100,
                c=df_subset['color'].iloc[0],
                alpha=0.7,
                edgecolors='white',
                linewidths=1,
                label=who_better,
                zorder=2
            )
        
        # Circle the extreme cases (smaller circles)
        for idx, case in enumerate(extreme_cases):
            circle = Circle(
                (case['human_error_rate'], case['ai_error_rate']),
                radius=2,  # Reduced from 8 to 4
                fill=False,
                edgecolor='black',
                linewidth=1.0,
                linestyle='--',
                zorder=3
            )
            ax_scatter.add_patch(circle)
            
            # Add label (A1, A2, B1, B2)
            label = f"{'A' if idx < 2 else 'B'}{(idx % 2) + 1}"
            ax_scatter.text(
                case['human_error_rate'] + 2.2,   # small x-offset so text appears to the right
                case['ai_error_rate'] + 0.1,
                label,
                ha='left',                      # anchor on the left so it extends to the right
                va='bottom',
                fontsize=12,
                fontweight='bold',
                zorder=4
            )
        ax_scatter.set_yticks([10, 90])
        ax_scatter.set_yticklabels(["LLM correct", "LLM wrong"])
        ax_scatter.set_xlabel('Human Error Rate in Percentage', fontsize=13, fontweight='bold')
        ax_scatter.set_ylabel('LLM Prediction', fontsize=13, fontweight='bold')
        ax_scatter.set_xlim(-5, 105)
        ax_scatter.set_ylim(-5, 105)
        ax_scatter.grid(True, alpha=0.3, linestyle='--')
        
        # Legend positioned at right middle
        ax_scatter.legend(
            loc='center right',
            fontsize=11,
            framealpha=0.9,
            bbox_to_anchor=(0.98, 0.5)
        )
        
        ax_scatter.set_title(
            f"{base_title}\nVideo {vid} - {video_names[vid]}",
            fontsize=15,
            fontweight='bold',
            pad=15
        )
       
        
        # Row 2: Four bar plots for extreme cases
        for idx, case in enumerate(extreme_cases):
            ax_bar = fig.add_subplot(gs[1, idx])
            
            # Prepare bar data with labels
            categories = [0, 1, -99]
            counts = [case['count_data'].get(cat, 0) for cat in categories]
            category_labels = [rating_labels[cat] for cat in categories]
            
            # Bar colors - all blue
            bar_colors = ['#1f77b4', '#1f77b4', '#1f77b4']
            
            # Create bars
            bars = ax_bar.bar(category_labels, counts, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add markers for AI prediction and reference (smaller markers)
            max_height = max(counts) if counts else 1
            marker_y = max_height + max(1, 0.05 * max_height)
            
            # AI prediction (diamond) - smaller
            ai_pred_idx = categories.index(case['ai_prediction']) if case['ai_prediction'] in categories else None
            if ai_pred_idx is not None:
                ax_bar.scatter(
                    ai_pred_idx,
                    marker_y,
                    s=120,  # Reduced from 200 to 120
                    marker='D',
                    color='red',
                    #edgecolors='black',
                    linewidths=1.5,
                    zorder=5
                )
            
            # Reference (star) - smaller
            ref_idx = categories.index(case['ref_rating']) if case['ref_rating'] in categories else None
            if ref_idx is not None:
                ax_bar.scatter(
                    ref_idx,
                    marker_y,
                    s=100,  # Reduced from 300 to 180
                    marker='*',
                    color='black',
                    #edgecolors='black',
                    linewidths=1.5,
                    zorder=5
                )
            
            # Labels and formatting
            label = f"{'A' if idx < 2 else 'B'}{(idx % 2) + 1}"
            ax_bar.set_title(
                f"{label}: {case['item']}",
                fontsize=10,
                fontweight='bold'
            )
            ax_bar.set_xlabel('Rating Category', fontsize=9)
            ax_bar.set_ylabel('Human Count', fontsize=9)
            ax_bar.set_ylim(0, marker_y * 1.15)
            ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
            ax_bar.tick_params(axis='x', labelsize=8)
            
            justification_text = case['ai_justification']
            wrapped_text = textwrap.fill(justification_text, width=50)
            ax_bar.text(
                0.5, -0.25,                           # x, y in axis coords
                f"LLM - Justification: \n{wrapped_text}",   # the justification string
                transform=ax_bar.transAxes,
                ha='center', va='top',
                fontsize=8, wrap=True, multialignment='center'
            )
        # Add shared legend for bar plots outside on the right
        # Create dummy handles for legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='AI prediction', linewidth=0),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
                markersize=12, label='Reference', linewidth=0)
        ]
        
        # Position legend to the right of all bar plots
        fig.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(0.98, 0.25),
            fontsize=10,
            framealpha=0.9,
            title='Bar Plot Markers',
            title_fontsize=10
        )
        #set resolution really high for publication and show
        if save:
            plt.savefig(f"figure_{vid}.png", dpi=600, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Video {vid} - {video_names[vid]}")
        print(f"{'='*60}")
        print(f"Total items: {len(df_plot)}")
        print(f"\nExtreme Cases Selected:")
        for idx, case in enumerate(extreme_cases):
            label = f"{'A' if idx < 2 else 'B'}{(idx % 2) + 1}"
            print(f"  {label}: {case['item']}")
            print(f"      Human error: {case['human_error_rate_original']:.1f}%, AI error: {case['ai_error_rate_original']:.1f}%")
        
        figures.append(fig)
    
    return figures

def classify_from_plot_data(
    df: pd.DataFrame,
    high_error_threshold: float = 80.0,  # Note: already in percentage
    low_error_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Classify items by difficulty using pre-computed plot data.
    """
    df = df.copy()
    
    # LLM correctness (error rate 0 = correct, 100 = wrong)
    df['llm_correct'] = df['ai_error_rate_original'] < 50
    
    # Classify difficulty
    conditions = [
        (df['human_error_rate_original'] > high_error_threshold) & (~df['llm_correct']),
        (df['human_error_rate_original'] <= low_error_threshold) & (~df['llm_correct']),
        (df['human_error_rate_original'] > high_error_threshold) & (df['llm_correct']),
        (df['human_error_rate_original'] <= low_error_threshold) & (df['llm_correct']),
    ]
    choices = [
        'shared_difficulty',
        'llm_specific_difficulty', 
        'clinician_specific_difficulty',
        'low_difficulty'
    ]
    df['difficulty_type'] = np.select(conditions, choices, default='moderate_difficulty')
    
    return df
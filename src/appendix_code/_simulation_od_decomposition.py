# =============================================================================
# SIMULATION DECOMPOSITION: OD vs Non-OD Items
# =============================================================================
# Uses the existing simulation_results from the paired-clinician simulation.
# Decomposes: does AI supervision help more on non-OD items?
# Requires: simulation_results, video_simulation_results, psy_cols_reduced,
#           OD_ITEMS_REDUCED, reference, video_ids_in_ki_master, master_df

OD_RED = set(OD_ITEMS_REDUCED)  # from disagreement analysis cell
NON_OD_RED = [c for c in psy_cols_reduced if c not in OD_RED]

print(f"OD items: {len(OD_RED)}, Non-OD items: {len(NON_OD_RED)}, Total: {len(psy_cols_reduced)}")

# ── Re-run simulation accuracy decomposed by OD vs non-OD ────────────────────
# We'll use the raw simulation details (per clinician-pair) and recompute
# accuracy separately for OD and non-OD item subsets.

ref_dedup_sim = reference_reduced.drop_duplicates(subset="ID_Video")
ref_lk = {}
for _, rr in ref_dedup_sim.iterrows():
    ref_lk[int(rr["ID_Video"])] = rr

# Get AI majority ratings
ai_site = f"{best_ai}_majority"
ai_rows = master_df[master_df["site"] == ai_site]

sim_detail = simulation_results['simulation_details']

# For each video, compute: baseline clinician accuracy, AI-supervised accuracy
# separately for OD and non-OD items
vid_decomp = []
for vid in sorted(video_ids_in_ki_master):
    rs = ref_lk.get(vid)
    if rs is None:
        continue
    ai_row = ai_rows[ai_rows["video_id"] == vid]
    if len(ai_row) == 0:
        continue
    ai_s = ai_row.iloc[0]
    
    for grp_name, cols in [("OD", list(OD_RED)), ("Non-OD", NON_OD_RED)]:
        # AI accuracy on this subset
        ai_correct = 0
        n_items = 0
        for col in cols:
            if col not in ai_s.index or col not in rs.index:
                continue
            try:
                av = float(ai_s[col].iloc[0]) if hasattr(ai_s[col], 'iloc') else float(ai_s[col])
                rv = float(rs[col].iloc[0]) if hasattr(rs[col], 'iloc') else float(rs[col])
            except:
                continue
            if av != av or rv != rv:
                continue
            n_items += 1
            if av == rv:
                ai_correct += 1
        
        # Mean clinician accuracy on this subset
        hv = human_master[human_master["video_id"] == vid]
        clin_accs = []
        for _, cr in hv.iterrows():
            cc = 0
            ct = 0
            for col in cols:
                if col not in cr.index or col not in rs.index:
                    continue
                try:
                    cv_val = float(cr[col].iloc[0]) if hasattr(cr[col], 'iloc') else float(cr[col])
                    rv_val = float(rs[col].iloc[0]) if hasattr(rs[col], 'iloc') else float(rs[col])
                except:
                    continue
                if cv_val != cv_val or cv_val == 10000.0 or rv_val != rv_val:
                    continue
                ct += 1
                if cv_val == rv_val:
                    cc += 1
            if ct > 0:
                clin_accs.append(cc / ct)
        
        ai_acc_grp = ai_correct / n_items if n_items > 0 else 0
        clin_mean = np.mean(clin_accs) if clin_accs else 0
        clin_std = np.std(clin_accs) if clin_accs else 0
        
        vid_decomp.append({
            "Video": video_names_da.get(vid, f"V{vid}"),
            "Item group": grp_name,
            "N items": n_items,
            "AI accuracy": round(ai_acc_grp, 3),
            "Clinician mean acc.": round(clin_mean, 3),
            "Clinician SD": round(clin_std, 3),
            "Delta (AI - Clin)": round(ai_acc_grp - clin_mean, 3),
            "AI > Clinician": ai_acc_grp > clin_mean,
        })

df_decomp = pd.DataFrame(vid_decomp)
display(HTML("<h3>Simulation Decomposition: AI vs. Clinician Accuracy by Item Type</h3>"))
display(HTML("<p><i>AI = GPT-5.1 majority vote. Clinician = mean across all 108 clinicians. "
             "Delta > 0 means AI outperforms clinicians on that item subset.</i></p>"))
display(df_decomp)

# ── Summary across videos ────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: AI vs. Clinician by Item Type (averaged across videos)")
print("="*80)
for grp in ["OD", "Non-OD"]:
    sub = df_decomp[df_decomp["Item group"] == grp]
    ai_mean = sub["AI accuracy"].mean()
    clin_mean = sub["Clinician mean acc."].mean()
    delta_mean = sub["Delta (AI - Clin)"].mean()
    print(f"\n  {grp} items:")
    print(f"    AI mean accuracy:       {ai_mean:.3f}")
    print(f"    Clinician mean accuracy: {clin_mean:.3f}")
    print(f"    Mean Delta:             {delta_mean:+.3f}")
    print(f"    AI wins on {int(sub['AI > Clinician'].sum())}/{len(sub)} videos")

# ── Publication-ready figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

videos_list = sorted(video_ids_in_ki_master)
vid_labels = [video_names_da.get(v, f"V{v}") for v in videos_list]
x = np.arange(len(videos_list))
bw = 0.18

for i, (grp, color_ai, color_clin) in enumerate([("OD", "#e74c3c", "#f5b7b1"), ("Non-OD", "#2ecc71", "#abebc6")]):
    sub = df_decomp[df_decomp["Item group"] == grp].sort_values("Video")
    ai_vals = sub["AI accuracy"].tolist()
    clin_vals = sub["Clinician mean acc."].tolist()
    offset = -bw if grp == "OD" else bw
    
    b1 = ax.bar(x + offset - bw/2, ai_vals, bw, label=f"AI ({grp})", color=color_ai, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + offset + bw/2, clin_vals, bw, label=f"Clinician ({grp})", color=color_clin, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(vid_labels, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("AI vs. Clinician Accuracy: OD vs. Non-OD Items per Video", fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_folder_figures}/simulation_od_vs_nonod_decomposition.png", dpi=300, bbox_inches='tight')
plt.show()

df_decomp.to_excel(f"{output_folder_tables}/simulation_od_decomposition.xlsx", index=False)
print(f"\nSaved to {output_folder_tables}/simulation_od_decomposition.xlsx")

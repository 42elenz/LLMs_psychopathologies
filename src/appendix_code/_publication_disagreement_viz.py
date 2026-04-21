# =============================================================================
# PUBLICATION TABLE + FIGURE: Disagreement & Error Complementarity
# =============================================================================

# ── TABLE: Clean summary for paper ───────────────────────────────────────────
pub_rows = []
for item_grp, od_mask in [("OD items (n=9)", True), ("Non-OD items (n=91)", False), ("All items (n=100)", None)]:
    sub_all = df_da if od_mask is None else df_da[df_da["is_od"] == od_mask]
    sub_dis = df_disagree if od_mask is None else df_disagree[df_disagree["is_od"] == od_mask]
    n_total = len(sub_all)
    n_agree = int(sub_all["agree"].sum())
    n_dis = len(sub_dis)
    
    # Disagreement types
    n_a = int((sub_dis["disagree_type"] == "clin_rated_llm_na").sum())
    n_b = int((sub_dis["disagree_type"] == "clin_na_llm_rated").sum())
    n_c = int((sub_dis["disagree_type"] == "both_rated_different").sum())
    
    # Who is correct at disagreement
    llm_w = int(sub_dis["llm_correct"].sum())
    clin_w = int(sub_dis["clin_correct"].sum())
    
    pub_rows.append({
        "Item group": item_grp,
        "N observations": f"{n_total:,}",
        "Agreement rate": f"{n_agree/n_total*100:.1f}%",
        "N disagreements": f"{n_dis:,}",
        "Type A: Clin rated, LLM=N/A": f"{n_a} ({n_a/n_dis*100:.0f}%)" if n_dis else "-",
        "Type B: Clin=N/A, LLM rated": f"{n_b} ({n_b/n_dis*100:.0f}%)" if n_dis else "-",
        "Type C: Both rated differently": f"{n_c} ({n_c/n_dis*100:.0f}%)" if n_dis else "-",
        "LLM correct (at disagree.)": f"{llm_w/n_dis*100:.1f}%" if n_dis else "-",
        "Clinician correct (at disagree.)": f"{clin_w/n_dis*100:.1f}%" if n_dis else "-",
        "Net acc. change (pp)": f"{(llm_w-clin_w)/n_total*100:+.2f}" if n_total else "-",
    })

df_pub_table = pd.DataFrame(pub_rows)
display(HTML("<h3>Table: Clinician-LLM Disagreement Analysis (Reduced Scale)</h3>"))
display(HTML("<p><i>Each observation = one clinician x item x video. "
             "LLM = GPT-5.1 majority vote (T=0.5, with AMDP definitions). "
             "OD = observation-dependent items requiring visual assessment. "
             "Net acc. change = accuracy change if LLM resolves all disagreements.</i></p>"))
display(df_pub_table)
df_pub_table.to_excel(f"{output_folder_tables}/publication_disagreement_table.xlsx", index=False)

# ── FIGURE: 2-panel publication figure ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Stacked horizontal bar - disagreement composition
labels = ["OD items\n(n=9)", "Non-OD items\n(n=91)"]
od_d = df_disagree[df_disagree["is_od"]]
nod_d = df_disagree[~df_disagree["is_od"]]

type_a = [100.0*int((od_d["disagree_type"]=="clin_rated_llm_na").sum())/max(len(od_d),1),
          100.0*int((nod_d["disagree_type"]=="clin_rated_llm_na").sum())/max(len(nod_d),1)]
type_b = [100.0*int((od_d["disagree_type"]=="clin_na_llm_rated").sum())/max(len(od_d),1),
          100.0*int((nod_d["disagree_type"]=="clin_na_llm_rated").sum())/max(len(nod_d),1)]
type_c = [100.0*int((od_d["disagree_type"]=="both_rated_different").sum())/max(len(od_d),1),
          100.0*int((nod_d["disagree_type"]=="both_rated_different").sum())/max(len(nod_d),1)]

y = np.arange(len(labels))
h = 0.5
ax1.barh(y, type_a, h, label="Clin rated, LLM = N/A", color="#e74c3c", alpha=0.85)
ax1.barh(y, type_b, h, left=type_a, label="Clin = N/A, LLM rated", color="#3498db", alpha=0.85)
left2 = [type_a[i]+type_b[i] for i in range(2)]
ax1.barh(y, type_c, h, left=left2, label="Both rated, different", color="#9b59b6", alpha=0.85)

for i in range(2):
    if type_a[i] > 8:
        ax1.text(type_a[i]/2, y[i], f"{type_a[i]:.0f}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    if type_b[i] > 8:
        ax1.text(type_a[i]+type_b[i]/2, y[i], f"{type_b[i]:.0f}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    if type_c[i] > 8:
        ax1.text(left2[i]+type_c[i]/2, y[i], f"{type_c[i]:.0f}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')

ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=11)
ax1.set_xlabel("% of disagreements", fontsize=11)
ax1.set_xlim(0, 100)
ax1.set_title("(a) Disagreement Composition", fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.invert_yaxis()

# Panel (b): Grouped bar - who is correct when they disagree
categories = ["OD items", "Non-OD items"]
llm_pct = [100.0*int(od_d["llm_correct"].sum())/max(len(od_d),1),
           100.0*int(nod_d["llm_correct"].sum())/max(len(nod_d),1)]
clin_pct = [100.0*int(od_d["clin_correct"].sum())/max(len(od_d),1),
            100.0*int(nod_d["clin_correct"].sum())/max(len(nod_d),1)]
both_pct = [100.0 - llm_pct[i] - clin_pct[i] for i in range(2)]

x = np.arange(len(categories))
bw = 0.22
b1 = ax2.bar(x - bw, llm_pct, bw, label="LLM correct", color="#2ecc71", edgecolor="black", linewidth=0.6)
b2 = ax2.bar(x, clin_pct, bw, label="Clinician correct", color="#e67e22", edgecolor="black", linewidth=0.6)
b3 = ax2.bar(x + bw, both_pct, bw, label="Both wrong", color="#bdc3c7", edgecolor="black", linewidth=0.6)

for bars in [b1, b2, b3]:
    for bar in bars:
        ht = bar.get_height()
        if ht > 2:
            ax2.text(bar.get_x() + bar.get_width()/2, ht + 1.2, f"{ht:.0f}%",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_ylabel("% of disagreements", fontsize=11)
ax2.set_ylim(0, 95)
ax2.set_title("(b) Resolution: Who Is Correct\nWhen They Disagree?", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_folder_figures}/publication_disagreement_complementarity.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_folder_figures}/publication_disagreement_complementarity.pdf", bbox_inches='tight')
plt.show()

print(f"\nFigure saved to {output_folder_figures}/publication_disagreement_complementarity.png/.pdf")
print(f"Table saved to {output_folder_tables}/publication_disagreement_table.xlsx")

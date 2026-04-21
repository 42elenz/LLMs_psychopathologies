# =============================================================================
# DISAGREEMENT ANALYSIS (best model, majority vote, reduced scale)
# =============================================================================
# FIX: reference_reduced has DUPLICATE rows per video.
# We deduplicate to one row per video BEFORE the loop.
import re, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

OD_ITEMS_REDUCED = [c + "_reduced" for c in [
    "p59_perplexity_final", "p61_blunted_affect_final",
    "p77_affective_lability_final", "p78_affective_incontinence_final",
    "p79_affective_rigidity_final", "p83_motor_restlessness_final",
    "p84_parakinesia_final", "p85_mannerisms_final", "p86_histrionics_final",
]]
psy_cols_red = [c + "_reduced" for c in psy_cols]
video_names_da = {7: 'Mania', 8: 'Depression', 9: 'Schizophrenia'}

# LLM majority vote
_master_tmp = pd.concat([human_master, best_ki], ignore_index=True)
_ai_maj = helper.get_ai_majority_vote_with_justifications(
    master_df=_master_tmp, ai_site_key=best_ai,
    psy_cols_regex=PSY_REGEX_REDUCED, vid_col="video_id", site_col="site", id_col="ids"
)
llm_lookup = {}
for _, _r in _ai_maj.iterrows():
    _v = int(_r["video_id"])
    llm_lookup[_v] = {c: float(_r[c]) for c in psy_cols_red if c in _r.index and pd.notna(_r[c])}

# DEDUPLICATE reference_reduced: one Series per video
ref_dedup = reference_reduced.drop_duplicates(subset="ID_Video")
ref_lookup = {}
for _, _rr in ref_dedup.iterrows():
    ref_lookup[int(_rr["ID_Video"])] = _rr

print(f"reference_reduced rows: {len(reference_reduced)}, after dedup: {len(ref_dedup)}")
print(f"Videos in ref_lookup: {sorted(ref_lookup.keys())}")

# ── Build disagreement table ─────────────────────────────────────────────────
rows_out = []
for vid in video_ids_in_ki_master:
    if vid not in ref_lookup:
        continue
    rs = ref_lookup[vid]  # pandas Series — guaranteed scalar
    hv = human_master[human_master["video_id"] == vid]
    lv = llm_lookup.get(vid, {})

    for _, cr in hv.iterrows():
        cid = cr["id_code_v2"]
        for col in psy_cols_red:
            try:
                _tmp = cr[col]
                cv = float(_tmp.iloc[0]) if hasattr(_tmp, 'iloc') else float(_tmp)
            except (TypeError, ValueError, IndexError):
                continue
            if cv != cv or cv == 10000.0:  # NaN check
                continue
            llv = lv.get(col, float('nan'))
            if llv != llv:
                continue
            _tmp2 = rs[col] if col in rs.index else float('nan')
            rv = float(_tmp2.iloc[0]) if hasattr(_tmp2, 'iloc') else float(_tmp2)
            if rv != rv:
                continue

            ag = (cv == llv)
            if ag:
                dt = "agreement"
            elif llv == -99.0 and cv != -99.0:
                dt = "clin_rated_llm_na"
            elif cv == -99.0 and llv != -99.0:
                dt = "clin_na_llm_rated"
            else:
                dt = "both_rated_different"

            rows_out.append({
                "video_id": vid, "clinician_id": cid, "item": col,
                "is_od": col in OD_ITEMS_REDUCED,
                "clin_val": cv, "llm_val": llv, "ref_val": rv,
                "agree": ag, "disagree_type": dt,
                "clin_correct": (cv == rv), "llm_correct": (llv == rv),
            })

df_da = pd.DataFrame(rows_out)
df_disagree = df_da[~df_da["agree"]].copy()

print(f"\nTotal observations: {len(df_da):,}")
print(f"Agreements: {df_da['agree'].sum():,} ({df_da['agree'].mean()*100:.1f}%)")
print(f"Disagreements: {len(df_disagree):,} ({len(df_disagree)/len(df_da)*100:.1f}%)")

# ── Disagreement type frequencies ────────────────────────────────────────────
type_labels = {
    "clin_rated_llm_na": "Type A: Clinician rated, LLM = N/A",
    "clin_na_llm_rated": "Type B: Clinician = N/A, LLM rated",
    "both_rated_different": "Type C: Both rated, different values",
}
tc = df_disagree["disagree_type"].value_counts()
print("\nDisagreement types:")
for d, lab in type_labels.items():
    print(f"  {lab}: {int(tc.get(d, 0)):,} ({int(tc.get(d,0))/len(df_disagree)*100:.1f}%)")

sdt = []
for ol, om in [("OD items", True), ("Non-OD items", False)]:
    s = df_disagree[df_disagree["is_od"] == om]; ns = len(s)
    r = {"Item Group": ol, "N": ns}
    for d, lab in type_labels.items():
        n = int((s["disagree_type"] == d).sum())
        r[lab] = f"{n} ({n/ns*100:.1f}%)" if ns else "0"
    sdt.append(r)
df_type_summary = pd.DataFrame(sdt)
display(HTML("<h3>Disagreement Types: OD vs. Non-OD</h3>"))
display(df_type_summary)

vtr = []
for v in sorted(video_ids_in_ki_master):
    for ol, om in [("OD", True), ("Non-OD", False)]:
        s = df_disagree[(df_disagree["video_id"]==v)&(df_disagree["is_od"]==om)]; ns=len(s)
        r = {"Video": video_names_da[v], "Items": ol, "N": ns}
        for d in type_labels:
            n = int((s["disagree_type"]==d).sum())
            r[d] = f"{n} ({n/ns*100:.0f}%)" if ns else "0"
        vtr.append(r)
display(pd.DataFrame(vtr))

# ── Supervision resolution ───────────────────────────────────────────────────
rr = []
for ol, om in [("OD", True), ("Non-OD", False), ("All", None)]:
    for d, dl in list(type_labels.items()) + [("all", "ALL")]:
        s = df_disagree if om is None else df_disagree[df_disagree["is_od"]==om]
        if d != "all": s = s[s["disagree_type"]==d]
        n = len(s)
        if not n: continue
        nl=int(s["llm_correct"].sum()); nc=int(s["clin_correct"].sum())
        nb=int(((~s["llm_correct"])&(~s["clin_correct"])).sum())
        rr.append({"Items":ol,"Type":dl,"N":n,
            "LLM":f"{nl}({nl/n*100:.0f}%)","Clin":f"{nc}({nc/n*100:.0f}%)","Both wrong":f"{nb}({nb/n*100:.0f}%)",
            "AI acc":round(nl/n,3),"No acc":round(nc/n,3),"D":round((nl-nc)/n,3)})
df_resolution = pd.DataFrame(rr)
display(HTML("<h3>Supervision Resolution</h3>"))
display(df_resolution)

# ── Per-video ────────────────────────────────────────────────────────────────
vr = []
for v in sorted(video_ids_in_ki_master):
    for ol, om in [("OD", True), ("Non-OD", False)]:
        s = df_disagree[(df_disagree["video_id"]==v)&(df_disagree["is_od"]==om)]; n=len(s)
        if not n: continue
        nl=int(s["llm_correct"].sum()); nc=int(s["clin_correct"].sum())
        nb=int(((~s["llm_correct"])&(~s["clin_correct"])).sum())
        vr.append({"Video":video_names_da[v],"Items":ol,"N":n,"LLM":nl,"Clin":nc,"Both":nb,
            "AI":round(nl/n,3),"No":round(nc/n,3),"D":round((nl-nc)/n,3)})
df_vid_res = pd.DataFrame(vr)
display(HTML("<h3>Per-Video Resolution</h3>"))
display(df_vid_res)

# ── Accuracy gain decomposition ──────────────────────────────────────────────
print("\n" + "="*80)
tot = len(df_da)
for d, dl in type_labels.items():
    s = df_disagree[df_disagree["disagree_type"]==d]; n=len(s)
    if not n: continue
    lw=int(s["llm_correct"].sum()); cw=int(s["clin_correct"].sum()); net=lw-cw
    print(f"{dl}: N={n}, LLM={lw}, Clin={cw}, Net={'+' if net>=0 else ''}{net}, Contrib={net/tot*100:+.3f}pp")
tl=int(df_disagree["llm_correct"].sum()); tc2=int(df_disagree["clin_correct"].sum()); tn=tl-tc2
print(f"\nTOTAL: LLM={tl}, Clin={tc2}, Net={'+' if tn>=0 else ''}{tn}, Overall={tn/tot*100:+.3f}pp")

# ── Complementarity ──────────────────────────────────────────────────────────
ns = df_disagree[df_disagree["disagree_type"].isin(["clin_rated_llm_na","clin_na_llm_rated"])]
rs2 = df_disagree[df_disagree["disagree_type"]=="both_rated_different"]
nn=int(ns["llm_correct"].sum())-int(ns["clin_correct"].sum())
rn=int(rs2["llm_correct"].sum())-int(rs2["clin_correct"].sum())
tt=nn+rn
print(f"\nN/A types (A+B): net={'+' if nn>=0 else ''}{nn}" + (f" -> {abs(nn)/abs(tt)*100:.1f}%" if tt else ""))
print(f"Rated type (C): net={'+' if rn>=0 else ''}{rn}" + (f" -> {abs(rn)/abs(tt)*100:.1f}%" if tt else ""))

df_type_summary.to_excel(f"{output_folder_tables}/disagreement_types_od_vs_nonod.xlsx", index=False)
df_resolution.to_excel(f"{output_folder_tables}/supervision_resolution_by_type.xlsx", index=False)
df_vid_res.to_excel(f"{output_folder_tables}/supervision_resolution_per_video.xlsx", index=False)
print(f"\nSaved to {output_folder_tables}/")

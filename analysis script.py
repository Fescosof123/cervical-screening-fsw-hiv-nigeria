"""
Cervical screening within HIV care for female sex workers, KP-CARE 1, Nigeria.
Analysis script. Reads the raw RADET line list and writes every number used in
the manuscript to 'analysis results.xlsx' and 'analysis log.txt'.

Usage: python "analysis script.py" <path to RADET xlsx, 8 November 2024> <output folder> [<path to later RADET xlsx, 10 April 2026>]
Python 3.13, pandas 3.0, statsmodels 0.15, scipy 1.18, matplotlib 3.11
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
EXTRACT = pd.Timestamp("2024-11-08")   # date of the RADET extract
CENSOR = pd.Timestamp("2024-09-30")    # end of FY24 Q4, last ART status date
log = open(f"{OUT}/analysis log.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

# ---------------------------------------------------------------- load
if SRC.endswith(".pkl"):
    r = pd.read_pickle(SRC)
else:
    r = pd.read_excel(SRC, sheet_name="radet")
r.columns = [c.strip() for c in r.columns]
COL = dict(art="ART Start Date (yyyy-mm-dd)", scr="Date of Cervical Cancer Screening (yyyy-mm-dd)",
         typ="Cervical Cancer Screening Type", meth="Cervical Cancer Screening Method",
         res="Result of Cervical Cancer Screening", vl="Current Viral Load (c/ml)",
         tx="Date of Precancerous Lesions Treatment (yyyy-mm-dd)")
say("All records in RADET extract:", len(r))
f = r[r["Target group"] == "FSW"].copy()
say("FSW records:", len(f), "| all female:", (f.Sex == "Female").all())
say("Duplicate patient IDs among FSW:", f["Patient ID"].duplicated().sum())
n_minor = (f.Age < 18).sum()
f = f[f.Age >= 18].copy()
say("Excluded aged under 18:", n_minor, "| analysis population:", len(f))

# ---------------------------------------------------------------- variables
f["screened"] = f[COL["scr"]].notna().astype(int)
f["age_grp"] = pd.cut(f.Age, [17, 24, 34, 44, 120], labels=["18-24", "25-34", "35-44", "45+"])
yrs = (CENSOR - f[COL["art"]]).dt.days / 365.25
f["art_years"] = yrs
f["art_dur"] = pd.cut(yrs, [-1, 1, 3, 5, 100], right=False, labels=["<1", "1-2", "3-4", "5+"])
f["setting"] = f["ART Enrollment Setting"]
f["art_status"] = np.where(f["Current ART Status"].isin(["Active", "Active Restart"]), "On ART", "Not on ART")
f["art_status_detail"] = f["Current ART Status"]
f["vl"] = np.select([f[COL["vl"]].isna(), f[COL["vl"]] < 1000], ["No result", "Suppressed"], "Unsuppressed")
m = f["Months of ARV Refill"]
f["refill"] = np.select([m < 3, m < 6], ["<3 months", "3-5 months"], "6 months")
f["state"] = f["State"]
f["facility"] = f["Facility Name"].str.strip()
f["stage"] = f["Clinical Staging at Last Visit"].fillna("Not recorded")
say("Missing values: age", f.Age.isna().sum(), "| ART start", f[COL["art"]].isna().sum(),
    "| viral load", f[COL["vl"]].isna().sum(), "| clinical stage", f["Clinical Staging at Last Visit"].isna().sum(),
    "| refill", m.isna().sum())
say("Age median (IQR):", f.Age.median(), f.Age.quantile(.25), f.Age.quantile(.75))
say("Years on ART median (IQR):", round(yrs.median(), 1), round(yrs.quantile(.25), 1), round(yrs.quantile(.75), 1))
say("ART status detail:\n", f.art_status_detail.value_counts().to_string())
say("Regimen line:\n", f["Current Regimen Line"].value_counts().to_string())
say("Clinical stage:\n", f.stage.value_counts().to_string())

def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson"); return 100 * lo, 100 * hi
def exact(k, n):
    lo, hi = proportion_confint(k, n, method="beta"); return 100 * lo, 100 * hi

N, K = len(f), int(f.screened.sum())
say(f"Overall coverage: {K}/{N} = {100*K/N:.1f}% (95% CI {wilson(K,N)[0]:.1f} to {wilson(K,N)[1]:.1f})")

# ---------------------------------------------------------------- Table 1
VARS = [("Age group (years)", "age_grp", ["18-24", "25-34", "35-44", "45+"]),
        ("State", "state", sorted(f.state.unique())),
        ("ART enrolment setting", "setting", ["Community", "Facility"]),
        ("Years since ART start", "art_dur", ["<1", "1-2", "3-4", "5+"]),
        ("Current ART status", "art_status", ["On ART", "Not on ART"]),
        ("Latest viral load", "vl", ["Suppressed", "Unsuppressed", "No result"]),
        ("ARV refill length", "refill", ["<3 months", "3-5 months", "6 months"])]
rows = []
for lab, v, levels in VARS:
    for lv in levels:
        d = f[f[v] == lv]; n, k = len(d), int(d.screened.sum()); lo, hi = wilson(k, n)
        rows.append([lab, lv, n, round(100 * n / N, 1), k, round(100 * k / n, 1), round(lo, 1), round(hi, 1)])
rows.append(["Total", "", N, 100.0, K, round(100 * K / N, 1), *[round(x, 1) for x in wilson(K, N)]])
t1 = pd.DataFrame(rows, columns=["Characteristic", "Category", "N", "Column %", "Screened n", "Screened %", "CI low", "CI high"])
say("\nTABLE 1\n", t1.to_string(index=False))

# facility coverage
fac = f.groupby(["state", "facility"]).screened.agg(["size", "sum"]).reset_index()
fac.columns = ["State", "Facility", "N", "Screened"]
fac["Coverage %"] = (100 * fac.Screened / fac.N).round(1)
ci = fac.apply(lambda x: wilson(x.Screened, x.N), axis=1)
fac["CI low"] = [round(c[0], 1) for c in ci]; fac["CI high"] = [round(c[1], 1) for c in ci]
say("\nFACILITY COVERAGE\n", fac.to_string(index=False))
say("Facility coverage range:", fac["Coverage %"].min(), "to", fac["Coverage %"].max(),
    "| median", fac["Coverage %"].median())

# ---------------------------------------------------------------- models
f["age_grp"] = f.age_grp.astype(str); f["art_dur"] = f.art_dur.astype(str)
FORM = ("screened ~ C(age_grp, Treatment('25-34')) + C(art_dur, Treatment('1-2')) + C(setting, Treatment('Community'))"
        " + C(art_status, Treatment('On ART')) + C(vl, Treatment('Suppressed')) + C(refill, Treatment('6 months'))")
TERMS = [("Age group (years)", "age_grp", ["18-24", "25-34", "35-44", "45+"], "25-34"),
         ("Years since ART start", "art_dur", ["<1", "1-2", "3-4", "5+"], "1-2"),
         ("ART enrolment setting", "setting", ["Community", "Facility"], "Community"),
         ("Current ART status", "art_status", ["On ART", "Not on ART"], "On ART"),
         ("Latest viral load", "vl", ["Suppressed", "Unsuppressed", "No result"], "Suppressed"),
         ("ARV refill length", "refill", ["6 months", "3-5 months", "<3 months"], "6 months")]

def poisson_pr(data, formula, cluster=None):
    mod = smf.glm(formula, data=data, family=sm.families.Poisson())
    if cluster is None:
        return mod.fit(cov_type="HC1")
    return mod.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})

def extract(res, var, level):
    hits = [p for p in res.params.index if p.startswith(f"C({var},") and p.endswith(f"[T.{level}]")]
    p = hits[0]; b = res.params[p]; lo, hi = res.conf_int().loc[p]
    return np.exp(b), np.exp(lo), np.exp(hi), res.pvalues[p]

def pr_table(data, adj_formula, cluster=None, label=""):
    adj = poisson_pr(data, adj_formula, cluster)
    out = []
    for lab, v, levels, ref in TERMS:
        crude = poisson_pr(data, f"screened ~ C({v}, Treatment('{ref}'))", cluster)
        for lv in levels:
            if lv == ref:
                out.append([lab, lv, "1 (reference)", "", "1 (reference)", ""]); continue
            a = extract(crude, v, lv); b = extract(adj, v, lv)
            fmt = lambda e: f"{e[0]:.2f} ({e[1]:.2f} to {e[2]:.2f})"
            pf = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
            out.append([lab, lv, fmt(a), pf(a[3]), fmt(b), pf(b[3])])
    return pd.DataFrame(out, columns=["Characteristic", "Category", "Crude PR (95% CI)", "p", "Adjusted PR (95% CI)", "p adj"]), adj

# Main model: modified Poisson, facility fixed effects, robust (HC1) standard errors
main_form = FORM + " + C(facility)"
t2, main = pr_table(f, main_form)
say("\nTABLE 2 main model (facility fixed effects, robust SE), n =", int(main.nobs), "\n", t2.to_string(index=False))

# collinearity check: variance inflation factors from the full design matrix; the maximum is reported for the individual-level terms
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = main.model.exog; names = main.model.exog_names
vifs = {names[i]: variance_inflation_factor(X, i) for i in range(1, X.shape[1]) if "facility" not in names[i]}
say("Max VIF among individual-level terms:", round(max(vifs.values()), 2))

# Sensitivity 1: no facility terms, standard errors clustered on facility
s1, s1m = pr_table(f, FORM, cluster="facility")
say("\nSENSITIVITY 1 (no facility terms, cluster-robust SE on 19 facilities)\n", s1.to_string(index=False))
# Sensitivity 2: guideline-eligible women with time to be screened
e = f[(f.Age >= 25) & (f.art_status == "On ART") & (f.art_years >= 1)].copy()
ke, ne = int(e.screened.sum()), len(e)
say("Sensitivity 2 population with a refill under 3 months:", int((e.refill == "<3 months").sum()), "| with no viral load result:", int((e.vl == "No result").sum()))
say(f"\nSensitivity 2 population: {ne}; coverage {ke}/{ne} = {100*ke/ne:.1f}% ({wilson(ke,ne)[0]:.1f} to {wilson(ke,ne)[1]:.1f})")
TERMS_BAK = TERMS
TERMS = [("Age group (years)", "age_grp", ["25-34", "35-44", "45+"], "25-34"),
         ("Years since ART start", "art_dur", ["1-2", "3-4", "5+"], "1-2"),
         ("ART enrolment setting", "setting", ["Community", "Facility"], "Community"),
         ("Latest viral load", "vl", ["Suppressed", "Unsuppressed", "No result"], "Suppressed"),
         ("ARV refill length", "refill", ["6 months", "3-5 months", "<3 months"], "6 months")]
F2 = ("screened ~ C(age_grp, Treatment('25-34')) + C(art_dur, Treatment('1-2')) + C(setting, Treatment('Community'))"
      " + C(vl, Treatment('Suppressed')) + C(refill, Treatment('6 months')) + C(facility)")
s2, s2m = pr_table(e, F2)
say("SENSITIVITY 2 (aged 25+, on ART, 1+ years since ART start)\n", s2.to_string(index=False))
TERMS = TERMS_BAK
# Sensitivity 3: logistic regression, odds ratios, same covariates
lg = smf.logit(main_form, data=f).fit(disp=0)
rows = []
for lab, v, levels, ref in TERMS:
    for lv in levels:
        if lv == ref: continue
        hits = [p for p in lg.params.index if p.startswith(f"C({v},") and p.endswith(f"[T.{lv}]")][0]
        lo, hi = lg.conf_int().loc[hits]
        rows.append([lab, lv, f"{np.exp(lg.params[hits]):.2f} ({np.exp(lo):.2f} to {np.exp(hi):.2f})"])
s3 = pd.DataFrame(rows, columns=["Characteristic", "Category", "Adjusted OR (95% CI)"])
say("\nSENSITIVITY 3 (logistic, facility fixed effects)\n", s3.to_string(index=False))

# Between-facility variation: adjusted facility log odds, method-of-moments variance, median odds ratio
lg2 = smf.logit(FORM.replace("screened ~ ", "screened ~ C(facility) + ") + " - 1", data=f).fit(disp=0)
fp = [p for p in lg2.params.index if p.startswith("C(facility)")]
b = lg2.params[fp].values; v = np.diag(lg2.cov_params().loc[fp, fp].values)
w = 1 / v; bw = (w * b).sum() / w.sum(); Q = (w * (b - bw) ** 2).sum(); k = len(b)
tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
mor = np.exp(np.sqrt(2 * tau2) * 0.6745); icc = tau2 / (tau2 + np.pi ** 2 / 3)
say(f"\nBetween-facility variance (adjusted, log-odds scale): tau2 = {tau2:.3f}; MOR = {mor:.2f}; ICC = {icc:.3f}; Q = {Q:.0f} on {k-1} df")
# adjusted (standardised) facility coverage: predict everyone as if at each facility
std = []
for fa in sorted(f.facility.unique()):
    g = f.copy(); g["facility"] = fa
    std.append([fa, 100 * lg.predict(g).mean()])
std = pd.DataFrame(std, columns=["Facility", "Standardised coverage %"]).round(1)
fac = fac.merge(std, on="Facility")
say("Standardised facility coverage range:", fac["Standardised coverage %"].min(), "to", fac["Standardised coverage %"].max())
say(fac.to_string(index=False))

# ---------------------------------------------------------------- trend and timing
s = f[f.screened == 1].copy()
s["year"] = s[COL["scr"]].dt.year
say("\nScreening date range:", s[COL["scr"]].min().date(), "to", s[COL["scr"]].max().date())
say("Screenings by year:\n", s.year.value_counts().sort_index().to_string())
say("Screened after 30 Sep 2024:", (s[COL["scr"]] > CENSOR).sum())
trend = []
for y in [2021, 2022, 2023, 2024]:
    end = pd.Timestamp(f"{y}-12-31") if y < 2024 else EXTRACT
    den = int((f[COL["art"]] <= end).sum()); num = int((s[COL["scr"]] <= end).sum()); ny = int((s.year == y).sum())
    trend.append([y, ny, den, num, round(100 * num / den, 1)])
trend = pd.DataFrame(trend, columns=["Year", "Women screened in year", "Women started on ART by period end", "Cumulative screened", "Cumulative coverage %"])
say("\nTREND\n", trend.to_string(index=False))
gap = (s[COL["scr"]] - s[COL["art"]]).dt.days / 365.25
say("Years from ART start to screening, median (IQR):", round(gap.median(), 1), round(gap.quantile(.25), 1), round(gap.quantile(.75), 1),
    "| screened before ART start:", int((gap < 0).sum()), "| within 1 year of ART start:", int(((gap >= 0) & (gap < 1)).sum()))
say("Screening type:\n", s[COL["typ"]].value_counts().to_string())
say("Screening method:\n", s[COL["meth"]].value_counts().to_string())
say("Age <25 screened:", int((s.Age < 25).sum()), "of", int((f.Age < 25).sum()))

# ---------------------------------------------------------------- results and yield
s["has_result"] = s[COL["res"]].notna()
s["abnormal"] = s[COL["res"]].isin(["Positive", "Suspicious for cancer"])
nr = int(s.has_result.sum()); ns = len(s)
say(f"\nResult recorded: {nr}/{ns} = {100*nr/ns:.1f}%; no result: {ns-nr} = {100*(ns-nr)/ns:.1f}%")
say("Result values:\n", s[COL["res"]].value_counts(dropna=False).to_string())
s["days_since"] = (EXTRACT - s[COL["scr"]]).dt.days
old = s[s.days_since >= 90]
say(f"Screened 90+ days before extract: {len(old)}; no result {int((~old.has_result).sum())} = {100*(~old.has_result).mean():.1f}%")
rec = s[s.days_since < 90]
say(f"Screened <90 days before extract: {len(rec)}; no result {int((~rec.has_result).sum())} = {100*(~rec.has_result).mean():.1f}%")
def casc(d, lab, cat):
    n = len(d); nres = int(d.has_result.sum()); ab = int(d.abnormal.sum())
    pos = int((d[COL["res"]] == "Positive").sum()); sus = int((d[COL["res"]] == "Suspicious for cancer").sum())
    lo, hi = exact(ab, nres) if nres else (np.nan, np.nan)
    return [lab, cat, n, n - nres, round(100 * (n - nres) / n, 1), nres, pos, sus, ab,
            round(100 * ab / nres, 2) if nres else np.nan, round(lo, 2), round(hi, 2)]
rows = [casc(s, "All screened women", "")]
for y in sorted(s.year.unique()):
    rows.append(casc(s[s.year == y], "Year of screening", int(y)))
for mth in ["PAP Smear", "VIA", "HPV"]:
    rows.append(casc(s[s[COL["meth"]] == mth], "Screening method", mth))
for st in sorted(s.state.unique()):
    rows.append(casc(s[s.state == st], "State", st))
t3 = pd.DataFrame(rows, columns=["Group", "Category", "Screened", "No result n", "No result %", "With result", "Positive", "Suspicious for cancer", "Abnormal n", "Abnormal %", "CI low", "CI high"])
say("\nTABLE 3\n", t3.to_string(index=False))
facres = s.groupby(["state", "facility"]).agg(Screened=("has_result", "size"), With_result=("has_result", "sum"), Abnormal=("abnormal", "sum")).reset_index()
facres["No result %"] = (100 * (1 - facres.With_result / facres.Screened)).round(1)
facres["Abnormal % of results"] = (100 * facres.Abnormal / facres.With_result).round(2)
say("\nRESULTS BY FACILITY\n", facres.to_string(index=False))
ab = s[s.abnormal]
say("\nAbnormal results: n =", len(ab), "| age median (IQR):", ab.Age.median(), ab.Age.quantile(.25), ab.Age.quantile(.75))
say("Abnormal by ART status:\n", ab.art_status.value_counts().to_string())
say("Abnormal by viral load:\n", ab.vl.value_counts().to_string())
say("Abnormal with a precancer treatment date recorded:", int(ab[COL["tx"]].notna().sum()))
say("Any woman with a precancer treatment date:", int(f[COL["tx"]].notna().sum()),
    "| their results:", f.loc[f[COL["tx"]].notna(), COL["res"]].tolist())
say("Abnormal by screening type:\n", ab[COL["typ"]].value_counts().to_string())

# ---------------------------------------------------------------- follow-up in a later extract (optional third argument)
followup = None
if len(sys.argv) > 3:
    r2 = pd.read_pickle(sys.argv[3]) if sys.argv[3].endswith(".pkl") else pd.read_excel(sys.argv[3])
    r2.columns = [str(c).strip() for c in r2.columns]
    say("\nFOLLOW-UP EXTRACT: records", len(r2), "| latest pickup", r2["Last Pickup Date (yyyy-mm-dd)"].max().date(),
        "| latest screening date", pd.to_datetime(r2[COL["scr"]], errors="coerce").max().date(),
        "| target group blank", int(r2["Target group"].isna().sum()))
    k2 = r2[["Patient ID", COL["scr"], COL["res"], COL["tx"], "Current ART Status"]].drop_duplicates("Patient ID").rename(columns={"Current ART Status": "status_26"})
    mm = f[["Patient ID", "state", COL["scr"], COL["res"]]].merge(k2, on="Patient ID", how="left", suffixes=("_24", "_26"), indicator=True)
    bo = mm[mm._merge == "both"]; un = mm[mm._merge == "left_only"]
    say(f"Matched by client identifier: {len(bo)}/{len(mm)} = {100*len(bo)/len(mm):.1f}%; unmatched {len(un)}, by state:", un.state.value_counts().to_dict())
    ms = bo[bo[COL["scr"] + "_24"].notna() & bo[COL["res"] + "_24"].isna()]
    still = int(ms[COL["res"] + "_26"].isna().sum())
    say(f"Screened with no result in 2024, matched: {len(ms)}; still no result: {still} = {100*still/len(ms):.1f}%; results since entered:", ms[COL["res"] + "_26"].value_counts().to_dict())
    act26 = int((ms.status_26 == "Active").sum())
    say(f"Of these {len(ms)} women, active on ART in the later extract: {act26} = {100*act26/len(ms):.1f}%")
    ab2 = bo[bo[COL["res"] + "_24"].isin(["Positive", "Suspicious for cancer"])]
    say(f"Abnormal in 2024, matched: {len(ab2)}; result in later extract:", ab2[COL["res"] + "_26"].value_counts(dropna=False).to_dict(), "| with treatment date:", int(ab2[COL["tx"]].notna().sum()))
    ns2 = bo[bo[COL["scr"] + "_24"].isna()]
    say(f"Unscreened in 2024, matched: {len(ns2)}; with a screening date in later extract: {int(ns2[COL['scr'] + '_26'].notna().sum())}")
    followup = pd.DataFrame([["Women in the analysis", len(mm)], ["Matched to the later extract", len(bo)], ["Not matched", len(un)],
        ["Not matched, in Jigawa or Yobe", int(un.state.isin(["Jigawa", "Yobe"]).sum())],
        ["Screened with no result in 2024 (matched)", len(ms)], ["Of these, active on ART in the later extract", act26], ["Still no result in the later extract", still],
        ["Result since entered: negative", int((ms[COL["res"] + "_26"] == "Negative").sum())],
        ["Result since entered: positive or suspicious for cancer", int(ms[COL["res"] + "_26"].isin(["Positive", "Suspicious for cancer"]).sum())],
        ["Abnormal result in 2024 (matched)", len(ab2)], ["With a treatment date in the later extract", int(ab2[COL["tx"]].notna().sum())],
        ["Recorded as negative in the later extract", int((ab2[COL["res"] + "_26"] == "Negative").sum())]], columns=["Item", "n"])

# ---------------------------------------------------------------- figures
# Facilities are shown by state and a letter code; the key is in the results workbook.
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fo = fac.sort_values(["State", "Coverage %"]).reset_index(drop=True)
fo["Code"] = fo.groupby("State").cumcount().map(lambda i: "ABCDE"[i])
fo["Label"] = np.where(fo.groupby("State").State.transform("size") > 1, fo.State + " " + fo.Code, fo.State)
fac = fac.merge(fo[["Facility", "Label"]], on="Facility")
fig = plt.figure(figsize=(10.5, 6.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], wspace=0.28, hspace=0.38)
ax = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
ypos = np.arange(len(fo))[::-1]
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000", "#7F7F7F"]  # colour-blind-safe (Okabe-Ito based)
scol = {st: PAL[i] for i, st in enumerate(sorted(fo.State.unique()))}
for yy, (_, x) in zip(ypos, fo.iterrows()):
    ax.errorbar(x["Coverage %"], yy, xerr=[[x["Coverage %"] - x["CI low"]], [x["CI high"] - x["Coverage %"]]],
                fmt="o", color=scol[x.State], ms=6, capsize=2.5, lw=1.4)
ax.set_yticks(ypos); ax.set_yticklabels([f"{a} (n={n:,})" for a, n in zip(fo.Label, fo.N)])
ax.axvline(100 * K / N, ls="--", color="grey", lw=1)
ax.text(100 * K / N + 1, len(fo) - 0.45, f"All facilities {100*K/N:.1f}%", fontsize=8, color="0.3")
ax.set_xlabel("Women with a documented cervical screening (%)"); ax.set_xlim(0, 80)
ax.set_title("A", loc="left", fontweight="bold")
xs = trend.Year.astype(str)
a1.bar(xs, trend["Women screened in year"], color="#0072B2", edgecolor="black", linewidth=0.6)
for x, yv in zip(xs, trend["Women screened in year"]): a1.text(x, yv + 120, f"{yv:,}", ha="center", fontsize=8)
a1.set_ylabel("Women screened in the year"); a1.set_ylim(0, 7200); a1.set_title("B", loc="left", fontweight="bold")
a2.plot(xs, trend["Cumulative coverage %"], "-o", color="#D55E00", lw=2, ms=6)
for x, yv in zip(xs, trend["Cumulative coverage %"]): a2.text(x, yv + 2.2, f"{yv}%", ha="center", fontsize=8)
a2.set_ylabel("Cumulative coverage (%)"); a2.set_ylim(0, 40); a2.set_title("C", loc="left", fontweight="bold")
a2.set_xlabel("Calendar year")
for a in (ax, a1, a2): a.spines[["top", "right"]].set_visible(False)
for ext in ("jpg", "tif"):
    fig.savefig(f"{OUT}/Figure 1.{ext}", dpi=300, bbox_inches="tight", **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
fig.savefig(f"{OUT}/Figure 1.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- flow numbers and export
flow = pd.DataFrame([["Records in RADET extract (all key populations)", len(r)],
                     ["Female sex worker records", len(r[r['Target group'] == 'FSW'])],
                     ["Excluded: aged under 18 years", n_minor],
                     ["Analysis population", N], ["Documented cervical screening", K], ["No documented screening", N - K],
                     ["Screened, result recorded", nr], ["Screened, no result recorded", ns - nr],
                     ["Abnormal result (positive or suspicious for cancer)", int(s.abnormal.sum())]], columns=["Step", "n"])
say("\nFLOW\n", flow.to_string(index=False))
with pd.ExcelWriter(f"{OUT}/analysis results.xlsx") as xw:
    flow.to_excel(xw, "Flow", index=False); t1.to_excel(xw, "Table 1", index=False); t2.to_excel(xw, "Table 2", index=False)
    t3.to_excel(xw, "Table 3", index=False); fac.to_excel(xw, "Facility coverage", index=False); trend.to_excel(xw, "Trend", index=False)
    s1.to_excel(xw, "Sens 1 cluster SE", index=False); s2.to_excel(xw, "Sens 2 eligible", index=False); s3.to_excel(xw, "Sens 3 logistic", index=False)
    facres.to_excel(xw, "Results by facility", index=False)
    if followup is not None: followup.to_excel(xw, "Follow-up 2026", index=False)
log.close()

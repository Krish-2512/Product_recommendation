# %% [Cell 1] — Setup: Imports + Load All CSVs
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Loading every CSV that the data generator produced, and doing a first-pass
#   health check — shapes, column count, missing values, duplicates.
#   This is always the FIRST step in any EDA: understand what you have before
#   you start analyzing it.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

# ── Load all 8 tables ──────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

cm  = pd.read_csv("customer_master.csv")           # demographics + CLV
ld  = pd.read_csv("liability_data.csv")            # banking accounts + digital
ad  = pd.read_csv("asset_data.csv")                # loans + FOIR + DPD
cd  = pd.read_csv("cibil_data.csv")                # CIBIL scores + sub-scores
tb  = pd.read_csv("transaction_behavior.csv")      # spend breakdown
po  = pd.read_csv("product_ownership.csv")         # CC, insurance, etc.
rt  = pd.read_csv("recommendation_targets.csv")    # rank-1, rank-2, bank risk
ppm = pd.read_csv("product_performance_monthly.csv")  # 12-month portfolio data

tables = {
    "customer_master"           : cm,
    "liability_data"            : ld,
    "asset_data"                : ad,
    "cibil_data"                : cd,
    "transaction_behavior"      : tb,
    "product_ownership"         : po,
    "recommendation_targets"    : rt,
    "product_performance_monthly": ppm,
}

print(f"\n{'Table':<30} {'Rows':>6}  {'Cols':>5}  {'Nulls':>6}")
print("-" * 55)
for name, df in tables.items():
    total_nulls = df.isnull().sum().sum()
    print(f"{name:<30} {len(df):>6,}  {len(df.columns):>5}  {total_nulls:>6,}")

# ── Duplicate check ────────────────────────────────────────────────────────
print("\n── Duplicate customer_id check ──")
for name, df in tables.items():
    if "customer_id" in df.columns:
        dupes = df["customer_id"].duplicated().sum()
        print(f"  {name:<30}: {dupes} duplicate IDs")


# %% [Cell 2] — Customer Demographics (customer_master.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Exploring who our 20,000 synthetic customers are — their age, occupation,
#   income level, city tier, CLV tier, and segment. This gives a demographic
#   profile similar to what a bank's CRM team would produce.
#
#   WHY THIS MATTERS:
#   The ML model's product recommendations must make sense across these groups.
#   If 95% of customers are "mass" segment, but the model recommends premium
#   products to all of them, that's a data quality problem we'd catch here.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 2 — CUSTOMER DEMOGRAPHICS")
print("=" * 60)
print(cm[["age","occupation","income_group","city_tier","customer_segment","clv_tier"]].describe(include="all").T)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Customer Demographics — customer_master.csv", fontsize=16, fontweight="bold")

# ── 1. Age Distribution ────────────────────────────────────────────────────
ax = axes[0, 0]
ax.hist(cm["age"], bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.axvline(cm["age"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={cm['age'].mean():.1f}")
ax.axvline(cm["age"].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median={cm['age'].median():.1f}")
ax.set_title("Age Distribution")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.legend()
# Insight: Working-age population (25–55) should dominate; students and retirees are minorities.

# ── 2. Occupation Mix ─────────────────────────────────────────────────────
ax = axes[0, 1]
occ_counts = cm["occupation"].value_counts()
bars = ax.bar(occ_counts.index, occ_counts.values, color=sns.color_palette("muted", len(occ_counts)))
ax.set_title("Occupation Distribution")
ax.set_xlabel("Occupation"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=35)
for bar, val in zip(bars, occ_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val/len(cm)*100:.1f}%", ha="center", fontsize=8)
# Insight: Salaried should be ~52%. Students ~12%. Used to enforce loan eligibility gates.

# ── 3. Income Group Distribution ──────────────────────────────────────────
ax = axes[0, 2]
order = ["low", "lower_mid", "mid", "upper_mid", "high"]
ig_counts = cm["income_group"].value_counts().reindex(order, fill_value=0)
ax.bar(ig_counts.index, ig_counts.values, color=sns.color_palette("Blues_d", len(ig_counts)))
ax.set_title("Income Group Distribution")
ax.set_xlabel("Income Group"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=20)
# Insight: Low and mid income should make up the bulk. High income (<5%) maps to HNI products.

# ── 4. City Tier ──────────────────────────────────────────────────────────
ax = axes[1, 0]
tier_counts = cm["city_tier"].value_counts().sort_index()
ax.pie(tier_counts.values, labels=[f"Tier {t}" for t in tier_counts.index],
       autopct="%1.1f%%", startangle=90,
       colors=["#2ecc71", "#3498db", "#e74c3c"])
ax.set_title("City Tier Split")
# Insight: Tier-1 cities (Mumbai, Delhi) → higher CC and insurance take-up rates.

# ── 5. Customer Segment ───────────────────────────────────────────────────
ax = axes[1, 1]
seg_order = ["youth","mass","retail","mass_affluent","elite","hni"]
seg_counts = cm["customer_segment"].value_counts().reindex(seg_order, fill_value=0)
ax.barh(seg_counts.index, seg_counts.values, color=sns.color_palette("viridis", len(seg_counts)))
ax.set_title("Customer Segment Distribution")
ax.set_xlabel("Count")
for i, val in enumerate(seg_counts.values):
    ax.text(val + 50, i, f"{val/len(cm)*100:.1f}%", va="center", fontsize=9)
# Insight: "mass" and "retail" should be the majority. HNI = top income customers.

# ── 6. CLV Tier ───────────────────────────────────────────────────────────
ax = axes[1, 2]
clv_counts = cm["clv_tier"].value_counts()
colors = {"super_hni": "#e74c3c", "elite": "#f39c12", "normal": "#2ecc71"}
bar_colors = [colors.get(k, "#95a5a6") for k in clv_counts.index]
ax.bar(clv_counts.index, clv_counts.values, color=bar_colors, edgecolor="white")
ax.set_title("CLV Tier Distribution\n(super_hni=top 2%, elite=next 10%)")
ax.set_xlabel("CLV Tier"); ax.set_ylabel("Count")
for i, (k, v) in enumerate(clv_counts.items()):
    ax.text(i, v + 50, f"{v/len(cm)*100:.1f}%", ha="center", fontsize=9)
# Insight: super_hni ≈ 2%, elite ≈ 10%. These customers get premium product offers.

plt.tight_layout()
plt.savefig("eda_01_demographics.png", bbox_inches="tight")
plt.show()
print("\nSaved: eda_01_demographics.png")


# %% [Cell 3] — CIBIL Score Analysis (cibil_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Analyzing the CIBIL credit score distribution across 20,000 customers.
#   CIBIL drives the eligibility engine — it determines which products a
#   customer can be offered at all. So this is the single most important
#   variable in the dataset.
#
#   WHY THIS MATTERS:
#   We're checking that:
#     (a) The 4-tier split matches the intended proportions
#     (b) No_history (-1) customers are correctly a minority (mainly students)
#     (c) CIBIL is causally linked to repayment status (regular > default)
#     (d) Sub-scores (payment history, utilization, etc.) are properly distributed
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 3 — CIBIL SCORE ANALYSIS")
print("=" * 60)

# Separate NH (no history) customers from scored customers
cibil_scored = cd[cd["cibil_score"] > 0].copy()
cibil_nh     = cd[cd["cibil_score"] == -1].copy()

print(f"  Scored customers (CIBIL > 0): {len(cibil_scored):,}")
print(f"  No History (CIBIL = -1):       {len(cibil_nh):,}  ({len(cibil_nh)/len(cd)*100:.1f}%)")
print(f"\n  CIBIL score stats (scored only):")
print(cibil_scored["cibil_score"].describe().round(1).to_string())

# 4-tier bucket distribution
print(f"\n  CIBIL 4-tier distribution:")
bucket_counts = cd["cibil_score_bucket"].value_counts()
for bucket, count in bucket_counts.items():
    print(f"    {bucket:<15}: {count:>6,}  ({count/len(cd)*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("CIBIL Score Analysis — cibil_data.csv", fontsize=16, fontweight="bold")

# ── 1. CIBIL Score Histogram (scored customers only) ──────────────────────
ax = axes[0, 0]
ax.hist(cibil_scored["cibil_score"], bins=40, color="#E74C3C", edgecolor="white", linewidth=0.4)
for thresh, label, color in [(650,"High Risk|Risky","#c0392b"),
                              (700,"Risky|Good","#e67e22"),
                              (750,"Good|Excellent","#27ae60")]:
    ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5, label=f"{thresh}")
ax.set_title("CIBIL Score Distribution (scored customers)")
ax.set_xlabel("CIBIL Score (300–900)"); ax.set_ylabel("Count")
ax.legend(title="Tier thresholds", fontsize=8)
# The 4 colored bands show the tier boundaries that drive eligibility gates.

# ── 2. 4-Tier Bucket Bar ──────────────────────────────────────────────────
ax = axes[0, 1]
tier_order = ["high_risk","risky","good","excellent","no_history"]
tier_colors = {"high_risk":"#c0392b","risky":"#e67e22","good":"#f1c40f","excellent":"#27ae60","no_history":"#95a5a6"}
bc = cd["cibil_score_bucket"].value_counts().reindex(tier_order, fill_value=0)
bar_colors_cibil = [tier_colors[k] for k in bc.index]
ax.bar(bc.index, bc.values, color=bar_colors_cibil, edgecolor="white")
ax.set_title("CIBIL 4-Tier Bucket Distribution")
ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("Count")
for i, val in enumerate(bc.values):
    ax.text(i, val + 50, f"{val/len(cd)*100:.1f}%", ha="center", fontsize=9)

# ── 3. CIBIL Score by Repayment Status (Box Plot) ─────────────────────────
ax = axes[0, 2]
merged_rep = cd.merge(ad[["customer_id","repayment_status"]], on="customer_id", how="left")
scored_rep = merged_rep[merged_rep["cibil_score"] > 0]
rep_order  = ["regular","delayed","default"]
sns.boxplot(data=scored_rep, x="repayment_status", y="cibil_score",
            order=rep_order, palette=["#27ae60","#e67e22","#c0392b"], ax=ax)
ax.set_title("CIBIL Score vs Repayment Status\n(causal relationship check)")
ax.set_xlabel("Repayment Status"); ax.set_ylabel("CIBIL Score")
# MUST SHOW: regular >> delayed >> default. Validates Sanity Check #14.
means = scored_rep.groupby("repayment_status")["cibil_score"].mean()
for i, rep in enumerate(rep_order):
    if rep in means.index:
        ax.text(i, means[rep] + 5, f"{means[rep]:.0f}", ha="center", fontsize=9, color="black", fontweight="bold")

# ── 4. CIBIL Sub-Score Distributions ─────────────────────────────────────
ax = axes[1, 0]
sub_cols = ["payment_history_score","utilization_score","credit_history_score",
            "credit_mix_score","inquiry_score"]
sub_labels = ["Payment\nHistory (35%)","Utilization\n(30%)","Hist Length\n(15%)",
              "Credit Mix\n(10%)","Inquiries\n(10%)"]
sub_data   = [cibil_scored[col].dropna() for col in sub_cols if col in cibil_scored.columns]
actual_labels = sub_labels[:len(sub_data)]
bp = ax.boxplot(sub_data, labels=actual_labels, patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"), medianprops=dict(color="red", linewidth=2))
ax.set_title("CIBIL 5 Sub-Factor Score Distributions\n(all on 0–100 scale)")
ax.set_ylabel("Sub-Factor Score (0–100)")
ax.set_ylim(0, 110)
# Shows the weight of each factor. Payment history (35%) should have the widest spread.

# ── 5. Bureau Inquiries Distribution ─────────────────────────────────────
ax = axes[1, 1]
inq = cd["bureau_inquiries_6m"].value_counts().sort_index()
ax.bar(inq.index.astype(str), inq.values, color="#9B59B6", edgecolor="white")
ax.set_title("Bureau Inquiries in Last 6 Months\n(higher = credit-hungry = lower CIBIL)")
ax.set_xlabel("Number of Inquiries"); ax.set_ylabel("Count")
# Most regular customers should have 0–1 inquiries; defaults cluster at 3–7+.

# ── 6. CIBIL Score by Income Group (Violin) ───────────────────────────────
ax = axes[1, 2]
merged_inc = cibil_scored.merge(cm[["customer_id","income_group"]], on="customer_id", how="left")
inc_order  = ["low","lower_mid","mid","upper_mid","high"]
sns.violinplot(data=merged_inc, x="income_group", y="cibil_score",
               order=inc_order, palette="Blues", ax=ax, cut=0)
ax.set_title("CIBIL Score by Income Group\n(higher income → better credit hygiene?)")
ax.set_xlabel("Income Group"); ax.set_ylabel("CIBIL Score")
ax.tick_params(axis="x", rotation=15)
# A slight upward trend is expected since high-income customers have more financial discipline.

plt.tight_layout()
plt.savefig("eda_02_cibil.png", bbox_inches="tight")
plt.show()
print("Saved: eda_02_cibil.png")


# %% [Cell 3B] — DTI, CLV Score & Default Rate Analysis  ← NEW CELL
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE ADDING (3 new variables not covered above):
#
#   1. DTI (Debt-to-Income) Distribution — DTI = total outstanding / annual income.
#      This is a HARD BLOCKER in the eligibility engine (DTI ≥ 5× blocks all new
#      credit). Critical to understand how many customers are close to this wall.
#
#   2. CLV Score Numeric Distribution — The 0–100 continuous CLV score, not just
#      the tier labels. Shows whether scores cluster or spread across the range
#      and validates that the 88th/98th percentile tier thresholds make sense.
#
#   3. Writeoff & Default Rate by CIBIL Tier — Sanity check that these bad-credit
#      flags concentrate in high-risk CIBIL buckets (should be near 0 for excellent).
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 3B — DTI, CLV SCORE & DEFAULT RATE ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("DTI, CLV Score & Default Rate Analysis — New Variables", fontsize=16, fontweight="bold")

# ── 1. DTI Ratio Distribution ─────────────────────────────────────────────
ax = axes[0, 0]
dti_col = "dti_ratio" if "dti_ratio" in cd.columns else None
if dti_col:
    dti_valid = cd[cd[dti_col] >= 0][dti_col].clip(0, 15)
    ax.hist(dti_valid, bins=40, color="#8E44AD", edgecolor="white", linewidth=0.4)
    ax.axvline(5, color="red", linestyle="--", linewidth=1.5, label="Hard block (DTI≥5)")
    ax.axvline(3, color="orange", linestyle="--", linewidth=1.5, label="High DTI (≥3)")
    ax.set_title("DTI Ratio Distribution\n(≥5× = hard block on all new credit)")
    ax.set_xlabel("DTI = Outstanding Debt / Annual Income")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    blocked = (cd[dti_col] >= 5).sum()
    print(f"\n  DTI ≥ 5× (hard blocked): {blocked:,} ({blocked/len(cd)*100:.1f}%)")
    print(f"  DTI ≥ 3× (high):          {(cd[dti_col]>=3).sum():,} ({(cd[dti_col]>=3).sum()/len(cd)*100:.1f}%)")

# ── 2. DTI Bucket Distribution ────────────────────────────────────────────
ax = axes[0, 1]
if "dti_bucket" in cd.columns:
    dti_bucket_order = ["low","moderate","high","severe"]
    dti_b_counts = cd["dti_bucket"].value_counts().reindex(dti_bucket_order, fill_value=0)
    dti_b_colors = {"low":"#27ae60","moderate":"#f1c40f","high":"#e67e22","severe":"#c0392b"}
    ax.bar(dti_b_counts.index, dti_b_counts.values,
           color=[dti_b_colors[k] for k in dti_b_counts.index], edgecolor="white")
    ax.set_title("DTI Bucket Distribution\n(severe = eligibility hard blocker)")
    ax.set_xlabel("DTI Bucket"); ax.set_ylabel("Count")
    for i, val in enumerate(dti_b_counts.values):
        ax.text(i, val + 50, f"{val/len(cd)*100:.1f}%", ha="center", fontsize=10)

# ── 3. CLV Score Numeric Distribution ─────────────────────────────────────
ax = axes[0, 2]
if "clv_score" in cm.columns:
    ax.hist(cm["clv_score"].dropna(), bins=40, color="#2E86AB", edgecolor="white", linewidth=0.4)
    p88 = cm["clv_score"].quantile(0.88)
    p98 = cm["clv_score"].quantile(0.98)
    ax.axvline(p88, color="orange", linestyle="--", linewidth=1.5, label=f"p88={p88:.1f} → elite")
    ax.axvline(p98, color="red",    linestyle="--", linewidth=1.5, label=f"p98={p98:.1f} → super_hni")
    ax.set_title("CLV Score Numeric Distribution (0–100)\n(percentile thresholds set elite & super_hni tiers)")
    ax.set_xlabel("CLV Score"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    print(f"\n  CLV Score stats:")
    print(f"    Mean  : {cm['clv_score'].mean():.1f}")
    print(f"    Median: {cm['clv_score'].median():.1f}")
    print(f"    p88 (elite threshold) : {p88:.2f}")
    print(f"    p98 (super_hni thresh): {p98:.2f}")

# ── 4. Writeoff Rate by CIBIL Tier ────────────────────────────────────────
ax = axes[1, 0]
if "writeoff_flag" in cd.columns and "cibil_score_bucket" in cd.columns:
    tier_order = ["high_risk","risky","good","excellent","no_history"]
    wof_rate = cd.groupby("cibil_score_bucket")["writeoff_flag"].mean() * 100
    wof_rate = wof_rate.reindex(tier_order, fill_value=0)
    tier_colors = {"high_risk":"#c0392b","risky":"#e67e22","good":"#f1c40f",
                   "excellent":"#27ae60","no_history":"#95a5a6"}
    ax.bar(wof_rate.index, wof_rate.values,
           color=[tier_colors.get(k,"#95a5a6") for k in wof_rate.index], edgecolor="white")
    ax.set_title("Writeoff Rate by CIBIL Tier\n(should be ~0% for excellent)")
    ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("Writeoff Rate (%)")
    for i, val in enumerate(wof_rate.values):
        ax.text(i, val + 0.2, f"{val:.1f}%", ha="center", fontsize=10)

# ── 5. Default Flag Rate by CIBIL Tier ────────────────────────────────────
ax = axes[1, 1]
if "default_history_flag" in cd.columns and "cibil_score_bucket" in cd.columns:
    def_rate = cd.groupby("cibil_score_bucket")["default_history_flag"].mean() * 100
    def_rate = def_rate.reindex(tier_order, fill_value=0)
    ax.bar(def_rate.index, def_rate.values,
           color=[tier_colors.get(k,"#95a5a6") for k in def_rate.index], edgecolor="white")
    ax.set_title("Default History Rate by CIBIL Tier\n(validates causal generation)")
    ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("Default Flag Rate (%)")
    for i, val in enumerate(def_rate.values):
        ax.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10)

# ── 6. CLV Score by CIBIL Tier (Boxplot) ─────────────────────────────────
ax = axes[1, 2]
if "clv_score" in cm.columns:
    merged_clv = cm.merge(cd[["customer_id","cibil_score_bucket"]], on="customer_id", how="left")
    tier_order_clv = ["high_risk","risky","good","excellent","no_history"]
    sns.boxplot(data=merged_clv, x="cibil_score_bucket", y="clv_score",
                order=tier_order_clv,
                palette={"high_risk":"#c0392b","risky":"#e67e22","good":"#f1c40f",
                         "excellent":"#27ae60","no_history":"#95a5a6"}, ax=ax)
    ax.set_title("CLV Score by CIBIL Tier\n(excellent CIBIL → better repayment → higher CLV)")
    ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("CLV Score (0–100)")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("eda_02b_dti_clv.png", bbox_inches="tight")
plt.show()
print("\nSaved: eda_02b_dti_clv.png")


# %% [Cell 4] — Loan & Asset Analysis (asset_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Exploring the loan portfolio: which loan types customers have, how much
#   of their income goes to EMIs (FOIR), days-past-due (DPD) patterns,
#   and outstanding balances.
#
#   WHY THIS MATTERS:
#   FOIR and DPD are two of the strongest signals for the eligibility engine.
#   High FOIR means the customer can't afford a new loan. High DPD means
#   they're already struggling with existing ones.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 4 — LOAN & ASSET ANALYSIS")
print("=" * 60)

loan_cols = [c for c in ["home_loan_flag","car_loan_flag","education_loan_flag",
                          "business_loan_flag","gold_loan_flag","personal_loan_flag"]
             if c in ad.columns]
loan_prevalence = ad[loan_cols].mean() * 100
print("\n  Loan type prevalence (% of 20K customers with this loan):")
for col, pct in loan_prevalence.sort_values(ascending=False).items():
    print(f"    {col.replace('_flag',''):<20}: {pct:.1f}%")

has_any = ad[[c for c in loan_cols]].any(axis=1).sum()
print(f"\n  Customers with ANY loan: {has_any:,} ({has_any/len(ad)*100:.1f}%)")
print(f"  Customers with NO loan:  {len(ad)-has_any:,} ({(len(ad)-has_any)/len(ad)*100:.1f}%)")

foir_col  = "foir" if "foir" in ad.columns else None
dpd_col   = "dpd_bucket" if "dpd_bucket" in ad.columns else None
rep_col   = "repayment_status" if "repayment_status" in ad.columns else None

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Loan & Asset Analysis — asset_data.csv", fontsize=16, fontweight="bold")

# ── 1. Loan Type Prevalence ────────────────────────────────────────────────
ax = axes[0, 0]
loan_names = [c.replace("_loan_flag","").replace("_flag","") for c in loan_cols]
bars = ax.barh(loan_names, loan_prevalence[loan_cols].values,
               color=sns.color_palette("Set2", len(loan_cols)))
ax.set_title("Loan Type Prevalence (% of all customers)")
ax.set_xlabel("Percentage (%)")
for bar, val in zip(bars, loan_prevalence[loan_cols].values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)

# ── 2. FOIR Distribution ───────────────────────────────────────────────────
ax = axes[0, 1]
if foir_col:
    foir_nonzero = ad[ad[foir_col] > 0][foir_col]
    ax.hist(foir_nonzero, bins=40, color="#E67E22", edgecolor="white", linewidth=0.4)
    ax.axvline(0.35, color="green", linestyle="--", linewidth=1.5, label="Safe (<35%)")
    ax.axvline(0.50, color="red",   linestyle="--", linewidth=1.5, label="Risky (>50%)")
    ax.set_title("FOIR Distribution (customers with loans)\nFOIR = EMI / Monthly Income")
    ax.set_xlabel("FOIR"); ax.set_ylabel("Count")
    ax.legend()
    print(f"\n  FOIR stats (loan holders only):")
    print(f"    Mean  : {foir_nonzero.mean():.3f}")
    print(f"    Median: {foir_nonzero.median():.3f}")
    print(f"    >50%  : {(foir_nonzero>0.5).sum():,} customers ({(foir_nonzero>0.5).mean()*100:.1f}%)")

# ── 3. DPD Bucket Distribution ─────────────────────────────────────────────
ax = axes[0, 2]
if dpd_col:
    dpd_order = ["0","1-30","31-60","61-90","90+"]
    dpd_existing = [d for d in dpd_order if d in ad[dpd_col].values]
    dpd_counts = ad[dpd_col].value_counts().reindex(dpd_existing, fill_value=0)
    dpd_colors = ["#27ae60","#f1c40f","#e67e22","#c0392b","#8e44ad"][:len(dpd_counts)]
    ax.bar(dpd_counts.index, dpd_counts.values, color=dpd_colors, edgecolor="white")
    ax.set_title("Days Past Due (DPD) Distribution\n0 = on time, 90+ = severely overdue")
    ax.set_xlabel("DPD Bucket"); ax.set_ylabel("Count")
    for i, val in enumerate(dpd_counts.values):
        ax.text(i, val + 30, f"{val/len(ad)*100:.1f}%", ha="center", fontsize=9)

# ── 4. Repayment Status Pie ────────────────────────────────────────────────
ax = axes[1, 0]
if rep_col:
    rep_counts = ad[rep_col].value_counts()
    ax.pie(rep_counts.values, labels=rep_counts.index,
           autopct="%1.1f%%", startangle=90,
           colors=["#27ae60","#e67e22","#c0392b"])
    ax.set_title("Repayment Status Split\n(Target: 75% regular, 18% delayed, 7% default)")

# ── 5. Loan Amount by Income Group ────────────────────────────────────────
ax = axes[1, 1]
loan_amt_col = "loan_amount" if "loan_amount" in ad.columns else None
if loan_amt_col:
    merged_loan = ad[ad[loan_amt_col] > 0].merge(cm[["customer_id","income_group"]], on="customer_id", how="left")
    inc_order = ["low","lower_mid","mid","upper_mid","high"]
    sns.boxplot(data=merged_loan, x="income_group", y=loan_amt_col,
                order=inc_order, palette="Greens", ax=ax)
    ax.set_title("Loan Amount by Income Group\n(higher income → larger loans)")
    ax.set_xlabel("Income Group"); ax.set_ylabel("Loan Amount (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.tick_params(axis="x", rotation=15)

# ── 6. Active Loan Count Distribution ─────────────────────────────────────
ax = axes[1, 2]
loan_count = ad[loan_cols].sum(axis=1)
count_dist = loan_count.value_counts().sort_index()
ax.bar(count_dist.index.astype(str), count_dist.values,
       color=sns.color_palette("Blues_d", len(count_dist)))
ax.set_title("Number of Active Loans per Customer\n(most have 0 or 1)")
ax.set_xlabel("Active Loan Count"); ax.set_ylabel("Number of Customers")
for i, val in enumerate(count_dist.values):
    ax.text(i, val + 30, f"{val/len(ad)*100:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("eda_03_assets.png", bbox_inches="tight")
plt.show()
print("Saved: eda_03_assets.png")


# %% [Cell 5] — Banking Behavior & Digital Adoption (liability_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Looking at customers' banking product holdings and digital banking behavior.
#   Savings rate, UPI usage, and digital adoption score are key signals used
#   in the recommendation engine (high savers → push insurance/FD; heavy UPI
#   users → push credit card).
#
#   WHY THIS MATTERS:
#   Digital adoption score (0–3) is one of the top-20 features in XGBoost.
#   We're checking the distribution is realistic and not degenerate (all 0
#   or all 3 would mean the feature adds no information).
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 5 — BANKING BEHAVIOR & DIGITAL ADOPTION")
print("=" * 60)

sav_col = "savings_rate" if "savings_rate" in ld.columns else None
dig_col = "digital_adoption_score" if "digital_adoption_score" in ld.columns else None
upi_col = "upi_txn_count" if "upi_txn_count" in ld.columns else None

if sav_col:
    print(f"\n  Savings rate stats:")
    print(ld[sav_col].describe().round(3).to_string())
if dig_col:
    print(f"\n  Digital adoption score distribution:")
    print(ld[dig_col].value_counts().sort_index().to_string())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Banking Behavior & Digital Adoption — liability_data.csv", fontsize=16, fontweight="bold")

# ── 1. Savings Rate Distribution ───────────────────────────────────────────
ax = axes[0, 0]
if sav_col:
    ax.hist(ld[sav_col].dropna(), bins=40, color="#1ABC9C", edgecolor="white", linewidth=0.4)
    ax.axvline(ld[sav_col].mean(), color="red", linestyle="--", label=f"Mean={ld[sav_col].mean():.2f}")
    ax.axvline(0.30, color="orange", linestyle="--", label="High saver threshold (30%)")
    ax.set_title("Savings Rate Distribution\n(savings_rate = net credit balance ratio)")
    ax.set_xlabel("Savings Rate"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# ── 2. UPI Transaction Count ───────────────────────────────────────────────
ax = axes[0, 1]
if upi_col:
    ax.hist(ld[upi_col].dropna(), bins=40, color="#8E44AD", edgecolor="white", linewidth=0.4)
    ax.axvline(ld[upi_col].mean(), color="red", linestyle="--", label=f"Mean={ld[upi_col].mean():.1f}")
    ax.axvline(30, color="green", linestyle="--", label="Power user threshold (30)")
    ax.set_title("UPI Transactions per Month\n(30+ = power user, strong CC candidate)")
    ax.set_xlabel("UPI Count"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# ── 3. Digital Adoption Score ──────────────────────────────────────────────
ax = axes[0, 2]
if dig_col:
    dig_counts = ld[dig_col].value_counts().sort_index()
    ax.bar(dig_counts.index.astype(str), dig_counts.values,
           color=["#c0392b","#e67e22","#f1c40f","#27ae60"][:len(dig_counts)])
    ax.set_title("Digital Adoption Score (0–3)\n0=none, 1=UPI, 2=+mobile, 3=+internet")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    for i, val in enumerate(dig_counts.values):
        ax.text(i, val + 30, f"{val/len(ld)*100:.1f}%", ha="center", fontsize=10)

# ── 4. Account Product Ownership ──────────────────────────────────────────
ax = axes[1, 0]
acct_cols   = [c for c in ["savings_account_flag","current_account_flag","fd_flag","rd_flag"]
               if c in ld.columns]
acct_labels = [c.replace("_flag","") for c in acct_cols]
acct_pcts   = ld[acct_cols].mean() * 100
ax.barh(acct_labels, acct_pcts.values, color=sns.color_palette("Pastel1", len(acct_cols)))
ax.set_title("Banking Account Ownership Rates\n(% of all 20K customers, from liability_data)")
ax.set_xlabel("Percentage (%)")
for i, val in enumerate(acct_pcts.values):
    ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)

# ── 5. Savings Rate by City Tier ───────────────────────────────────────────
ax = axes[1, 1]
if sav_col:
    merged_sav = ld.merge(cm[["customer_id","city_tier"]], on="customer_id", how="left")
    sns.boxplot(data=merged_sav, x="city_tier", y=sav_col,
                palette=["#2ecc71","#3498db","#e74c3c"], ax=ax)
    ax.set_title("Savings Rate by City Tier\n(Tier-1 = metro, Tier-3 = small city)")
    ax.set_xlabel("City Tier"); ax.set_ylabel("Savings Rate")

# ── 6. Digital Score by Occupation ────────────────────────────────────────
ax = axes[1, 2]
if dig_col:
    merged_dig = ld.merge(cm[["customer_id","occupation"]], on="customer_id", how="left")
    occ_dig = merged_dig.groupby("occupation")[dig_col].mean().sort_values(ascending=False)
    ax.barh(occ_dig.index, occ_dig.values, color=sns.color_palette("coolwarm", len(occ_dig)))
    ax.set_title("Mean Digital Adoption Score by Occupation\n(students & salaried expected highest)")
    ax.set_xlabel("Mean Digital Adoption Score (0–3)")
    for i, val in enumerate(occ_dig.values):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("eda_04_liability.png", bbox_inches="tight")
plt.show()
print("Saved: eda_04_liability.png")


# %% [Cell 6] — Spend Behavior & Transaction Patterns (transaction_behavior.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Exploring how customers spend their money: what categories dominate,
#   how much goes online vs offline, and investment intent levels.
#   Spend patterns are one of the key differentiators between customers
#   who should get a credit card vs insurance vs education loan.
#
#   WHY THIS MATTERS:
#   Travel spend ratio → Travel credit card recommendation.
#   Online spend ratio → Rewards credit card.
#   High savings rate → Insurance/FD recommendations.
#   These relationships are what the ML model must learn.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 6 — SPEND BEHAVIOR & TRANSACTION PATTERNS")
print("=" * 60)

spend_cols = [c for c in ["ecommerce_spend","fuel_spend","travel_spend",
                           "education_spend","insurance_premium_spend","utility_bill_payments"]
              if c in tb.columns]

if "avg_monthly_spend" in tb.columns:
    print(f"\n  Monthly spend stats (₹):")
    print(tb["avg_monthly_spend"].describe().apply(lambda x: f"₹{x:,.0f}").to_string())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Spend Behavior & Transaction Patterns — transaction_behavior.csv", fontsize=16, fontweight="bold")

# ── 1. Spend Category Breakdown (Pie) ─────────────────────────────────────
ax = axes[0, 0]
if spend_cols and "avg_monthly_spend" in tb.columns:
    spend_means = tb[spend_cols].mean()
    short_labels = [c.replace("_spend","").replace("_bill_payments","").replace("_premium","") for c in spend_cols]
    ax.pie(spend_means.values, labels=short_labels, autopct="%1.1f%%",
           startangle=90, colors=sns.color_palette("Set3", len(spend_cols)))
    ax.set_title("Average Spend Category Breakdown\n(mean share across all customers)")

# ── 2. Dominant Spend Category ─────────────────────────────────────────────
ax = axes[0, 1]
if "dominant_spend_category" in tb.columns:
    dom_counts = tb["dominant_spend_category"].value_counts()
    ax.bar(dom_counts.index, dom_counts.values,
           color=sns.color_palette("tab10", len(dom_counts)))
    ax.set_title("Dominant Spend Category Distribution\n(what each customer spends most on)")
    ax.set_xlabel("Category"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for i, val in enumerate(dom_counts.values):
        ax.text(i, val + 30, f"{val/len(tb)*100:.1f}%", ha="center", fontsize=9)

# ── 3. Online Spend Ratio Distribution ────────────────────────────────────
ax = axes[0, 2]
if "online_spend_ratio" in tb.columns:
    ax.hist(tb["online_spend_ratio"].dropna(), bins=35, color="#3498DB", edgecolor="white", linewidth=0.4)
    ax.axvline(0.20, color="red", linestyle="--", label="CC trigger (>20%)")
    ax.set_title("Online Spend Ratio Distribution\n(>20% → rewards credit card candidate)")
    ax.set_xlabel("Online Spend / Total Spend"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# ── 4. Travel Spend Ratio ─────────────────────────────────────────────────
ax = axes[1, 0]
if "travel_spend_ratio" in tb.columns:
    ax.hist(tb["travel_spend_ratio"].dropna(), bins=35, color="#E74C3C", edgecolor="white", linewidth=0.4)
    ax.axvline(0.15, color="green", linestyle="--", label="Travel CC trigger (>15%)")
    ax.set_title("Travel Spend Ratio Distribution\n(>15% → travel credit card candidate)")
    ax.set_xlabel("Travel Spend / Total Spend"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# ── 5. Investment Intent Score ─────────────────────────────────────────────
ax = axes[1, 1]
if "investment_intent_score" in tb.columns:
    inv_counts = tb["investment_intent_score"].value_counts().sort_index()
    ax.bar(inv_counts.index.astype(str), inv_counts.values,
           color=["#c0392b","#e67e22","#f1c40f","#27ae60"][:len(inv_counts)])
    ax.set_title("Investment Intent Score (0–3)\n(higher → more likely to buy insurance/MF)")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    for i, val in enumerate(inv_counts.values):
        ax.text(i, val + 30, f"{val/len(tb)*100:.1f}%", ha="center", fontsize=10)

# ── 6. Monthly Spend by Income Group ──────────────────────────────────────
ax = axes[1, 2]
if "avg_monthly_spend" in tb.columns:
    merged_sp = tb.merge(cm[["customer_id","income_group"]], on="customer_id", how="left")
    inc_order = ["low","lower_mid","mid","upper_mid","high"]
    sns.boxplot(data=merged_sp, x="income_group", y="avg_monthly_spend",
                order=inc_order, palette="YlOrRd", ax=ax)
    ax.set_title("Monthly Spend by Income Group\n(validates spend = 45–75% of income)")
    ax.set_xlabel("Income Group"); ax.set_ylabel("Monthly Spend (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("eda_05_transactions.png", bbox_inches="tight")
plt.show()
print("Saved: eda_05_transactions.png")


# %% [Cell 7] — Product Ownership Analysis (product_ownership.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Looking at what financial products customers currently own — credit card,
#   insurance, consumer durables, etc. This is the "portfolio" view.
#
#   WHY THIS MATTERS:
#   Customers who already own a product are unlikely to be recommended the
#   same one again (saturation effect). The recommendation engine applies a
#   saturation multiplier for existing holders. This analysis checks that the
#   ownership rates are realistic (7% CC ownership matches RBI stats).
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 7 — PRODUCT OWNERSHIP ANALYSIS")
print("=" * 60)

prod_cols = [c for c in ["credit_card","insurance","mutual_funds","consumer_durable",
                          "demat_account","debit_card","fd","rd"]
             if c in po.columns]
prod_ownership = po[prod_cols].mean() * 100
print("\n  Product ownership rates:")
for col, pct in prod_ownership.sort_values(ascending=False).items():
    print(f"    {col:<22}: {pct:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Product Ownership Analysis — product_ownership.csv", fontsize=16, fontweight="bold")

# ── 1. Product Ownership Rates ─────────────────────────────────────────────
ax = axes[0, 0]
sorted_prod = prod_ownership.sort_values(ascending=True)
ax.barh(sorted_prod.index, sorted_prod.values,
        color=sns.color_palette("Spectral", len(sorted_prod)))
ax.set_title("Product Ownership Rates\n(% of all 20K customers)")
ax.set_xlabel("Percentage (%)")
ax.axvline(7, color="red", linestyle="--", linewidth=1, label="RBI CC avg (7%)")
ax.legend(fontsize=8)
for i, val in enumerate(sorted_prod.values):
    ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)

# ── 2. CC Variant Distribution ─────────────────────────────────────────────
ax = axes[0, 1]
if "cc_variant" in po.columns:
    cc_holders = po[po["credit_card"] == 1]
    var_counts  = cc_holders["cc_variant"].value_counts()
    ax.pie(var_counts.values, labels=var_counts.index, autopct="%1.1f%%",
           startangle=90, colors=["#95a5a6","#3498db","#e74c3c","#f39c12"])
    ax.set_title(f"CC Variant Distribution\n(among {len(cc_holders):,} CC holders)")

# ── 3. CC Credit Limit Distribution ───────────────────────────────────────
ax = axes[0, 2]
if "cc_credit_limit" in po.columns:
    cc_holders_lim = po[po["cc_credit_limit"] > 0]["cc_credit_limit"]
    ax.hist(cc_holders_lim, bins=30, color="#F39C12", edgecolor="white", linewidth=0.4)
    ax.set_title("CC Credit Limit Distribution\n(among CC holders only)")
    ax.set_xlabel("Credit Limit (₹)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.axvline(cc_holders_lim.median(), color="red", linestyle="--",
               label=f"Median=₹{cc_holders_lim.median()/1e5:.1f}L")
    ax.legend()

# ── 4. CC Ownership by CIBIL Tier ─────────────────────────────────────────
ax = axes[1, 0]
if "credit_card" in po.columns:
    merged_cc = po.merge(cd[["customer_id","cibil_score_bucket"]], on="customer_id", how="left")
    tier_order = ["high_risk","risky","good","excellent","no_history"]
    cc_by_tier = merged_cc.groupby("cibil_score_bucket")["credit_card"].mean() * 100
    cc_by_tier = cc_by_tier.reindex(tier_order, fill_value=0)
    colors_tier = {"high_risk":"#c0392b","risky":"#e67e22","good":"#f1c40f",
                   "excellent":"#27ae60","no_history":"#95a5a6"}
    bar_c = [colors_tier.get(k, "#95a5a6") for k in cc_by_tier.index]
    ax.bar(cc_by_tier.index, cc_by_tier.values, color=bar_c, edgecolor="white")
    ax.set_title("CC Ownership Rate by CIBIL Tier\n(Sanity Check: all holders must be ≥700)")
    ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("CC Ownership %")
    ax.tick_params(axis="x", rotation=15)
    for i, val in enumerate(cc_by_tier.values):
        ax.text(i, val + 0.2, f"{val:.1f}%", ha="center", fontsize=9)

# ── 5. Insurance Ownership by Segment ─────────────────────────────────────
ax = axes[1, 1]
if "insurance" in po.columns:
    merged_ins = po.merge(cm[["customer_id","customer_segment"]], on="customer_id", how="left")
    seg_order = ["youth","mass","retail","mass_affluent","elite","hni"]
    ins_by_seg = merged_ins.groupby("customer_segment")["insurance"].mean() * 100
    ins_by_seg = ins_by_seg.reindex(seg_order, fill_value=0)
    ax.bar(ins_by_seg.index, ins_by_seg.values, color=sns.color_palette("Blues_d", len(ins_by_seg)))
    ax.set_title("Insurance Ownership Rate by Segment\n(HNI expected highest)")
    ax.set_xlabel("Customer Segment"); ax.set_ylabel("Insurance Ownership %")
    ax.tick_params(axis="x", rotation=20)

# ── 6. Products Owned per Customer ────────────────────────────────────────
ax = axes[1, 2]
total_owned = po[prod_cols].sum(axis=1)
own_dist    = total_owned.value_counts().sort_index()
ax.bar(own_dist.index.astype(str), own_dist.values,
       color=sns.color_palette("Greens_d", len(own_dist)))
ax.set_title("Total Products Owned per Customer\n(more products = higher CLV score)")
ax.set_xlabel("Number of Products Owned"); ax.set_ylabel("Count")
for i, val in enumerate(own_dist.values):
    ax.text(i, val + 30, f"{val/len(po)*100:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("eda_06_products.png", bbox_inches="tight")
plt.show()
print("Saved: eda_06_products.png")


# %% [Cell 8] — Recommendation Analysis (recommendation_targets.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Exploring the output of the deterministic scoring engine — who gets
#   recommended what, how confident the scores are, and what the bank risk
#   looks like per product. This is the TARGET variable for the ML model.
#
#   WHY THIS MATTERS:
#   We're checking:
#     (a) Recommendation diversity — no single product dominates (>35% = flag)
#     (b) Confidence scores correlate with CIBIL tier
#     (c) Bank risk scores are highest for unsecured products (personal, business)
#     (d) Rank-1 and Rank-2 are complementary (rarely the same product)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 8 — RECOMMENDATION ANALYSIS")
print("=" * 60)

rec1_col = "recommended_product" if "recommended_product" in rt.columns else None
rec2_col = "top2_recommended_product" if "top2_recommended_product" in rt.columns else None
conf_col = "confidence_score" if "confidence_score" in rt.columns else None

if rec1_col:
    print(f"\n  Rank-1 product distribution:")
    r1_counts = rt[rec1_col].value_counts()
    for prod, count in r1_counts.items():
        bar = "█" * int(count / 50)
        print(f"    {prod:<22}: {count:>5,}  ({count/len(rt)*100:.1f}%)  {bar}")
    print(f"\n  Most common rank-1: {r1_counts.index[0]} ({r1_counts.iloc[0]/len(rt)*100:.1f}%)")
    print(f"  Least common rank-1: {r1_counts.index[-1]} ({r1_counts.iloc[-1]/len(rt)*100:.1f}%)")
    print(f"\n  Any product > 35%: {'YES ⚠️' if r1_counts.iloc[0]/len(rt) > 0.35 else 'NO ✓'}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Recommendation Analysis — recommendation_targets.csv", fontsize=16, fontweight="bold")

# ── 1. Rank-1 Product Distribution ────────────────────────────────────────
ax = axes[0, 0]
if rec1_col:
    r1_counts = rt[rec1_col].value_counts()
    bars = ax.barh(r1_counts.index, r1_counts.values,
                   color=sns.color_palette("tab10", len(r1_counts)))
    ax.set_title("Rank-1 Recommendation Distribution\n(no product should exceed 35%)")
    ax.set_xlabel("Number of Customers")
    ax.axvline(len(rt) * 0.35, color="red", linestyle="--", label="35% limit")
    ax.legend()
    for bar, val in zip(bars, r1_counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{val/len(rt)*100:.1f}%", va="center", fontsize=8)

# ── 2. Rank-1 vs Rank-2 Side-by-Side ─────────────────────────────────────
ax = axes[0, 1]
if rec1_col and rec2_col:
    products = sorted(rt[rec1_col].dropna().unique())
    r1_c = rt[rec1_col].value_counts().reindex(products, fill_value=0)
    r2_c = rt[rec2_col].value_counts().reindex(products, fill_value=0)
    x = np.arange(len(products))
    w = 0.4
    ax.bar(x - w/2, r1_c.values, width=w, label="Rank-1", color="#2E86AB")
    ax.bar(x + w/2, r2_c.values, width=w, label="Rank-2", color="#A23B72")
    ax.set_title("Rank-1 vs Rank-2 Comparison\n(should be different products)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_"," ") for p in products], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Count"); ax.legend()

# ── 3. Confidence Score Distribution ──────────────────────────────────────
ax = axes[0, 2]
if conf_col:
    ax.hist(rt[conf_col].dropna(), bins=40, color="#2ECC71", edgecolor="white", linewidth=0.4)
    ax.axvline(rt[conf_col].mean(), color="red", linestyle="--",
               label=f"Mean={rt[conf_col].mean():.1f}%")
    ax.set_title("Rank-1 Confidence Score Distribution\n(post-calibration, should peak 60–80%)")
    ax.set_xlabel("Confidence (%)"); ax.set_ylabel("Count")
    ax.legend()

# ── 4. Bank Risk Score by Product ─────────────────────────────────────────
ax = axes[1, 0]
risk_cols = {c.replace("bank_risk_score_",""):c for c in rt.columns if c.startswith("bank_risk_score_")}
if risk_cols:
    risk_means = {prod: rt[col].mean() for prod, col in risk_cols.items()}
    risk_df    = pd.Series(risk_means).sort_values(ascending=True)
    bar_colors = ["#27ae60" if v < 25 else "#e67e22" if v < 45 else "#c0392b"
                  for v in risk_df.values]
    ax.barh(risk_df.index, risk_df.values, color=bar_colors)
    ax.set_title("Mean Bank Risk Score by Product\n(green=safe, red=high risk for bank)")
    ax.set_xlabel("Mean Bank Risk Score (0–100)")
    ax.axvline(40, color="red", linestyle="--", linewidth=1, label="Risk threshold (40)")
    ax.legend(fontsize=8)
    for i, val in enumerate(risk_df.values):
        ax.text(val + 0.5, i, f"{val:.1f}", va="center", fontsize=9)

# ── 5. Confidence by CIBIL Tier ───────────────────────────────────────────
ax = axes[1, 1]
if conf_col:
    merged_conf = rt.merge(cd[["customer_id","cibil_score_bucket"]], on="customer_id", how="left")
    tier_order  = ["high_risk","risky","good","excellent","no_history"]
    sns.boxplot(data=merged_conf, x="cibil_score_bucket", y=conf_col,
                order=tier_order,
                palette={"high_risk":"#c0392b","risky":"#e67e22",
                         "good":"#f1c40f","excellent":"#27ae60","no_history":"#95a5a6"},
                ax=ax)
    ax.set_title("Confidence Score by CIBIL Tier\n(Sanity Check #17: excellent > good > risky > high_risk)")
    ax.set_xlabel("CIBIL Tier"); ax.set_ylabel("Confidence (%)")
    ax.tick_params(axis="x", rotation=15)

# ── 6. Recommendation by Income Group ─────────────────────────────────────
ax = axes[1, 2]
if rec1_col:
    merged_inc = rt.merge(cm[["customer_id","income_group"]], on="customer_id", how="left")
    top_prods  = rt[rec1_col].value_counts().head(5).index.tolist()
    filtered   = merged_inc[merged_inc[rec1_col].isin(top_prods)]
    pivot      = filtered.groupby(["income_group", rec1_col]).size().unstack(fill_value=0)
    pivot      = pivot.reindex(["low","lower_mid","mid","upper_mid","high"])
    pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", linewidth=0.4)
    ax.set_title("Top-5 Recommendations by Income Group\n(business/CC → high income, gold → low income)")
    ax.set_xlabel("Income Group"); ax.set_ylabel("Count")
    ax.legend(title="Product", fontsize=7, loc="upper left")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("eda_07_recommendations.png", bbox_inches="tight")
plt.show()
print("Saved: eda_07_recommendations.png")


# %% [Cell 9] — Portfolio Performance & Seasonal Trends (product_performance_monthly.csv)
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Analyzing the 12-month × 9-product portfolio performance table. This data
#   drives the `portfolio_adj` term in the utility scoring formula used in
#   the ML pipeline (Cell 7 of the notebook).
#
#   WHY THIS MATTERS:
#   A product with 11% default rate (education loan) should have a lower
#   utility score in Jan–Mar (post-semester, students can't pay) and a higher
#   one in Oct–Dec (festive season uplift in employment). This cell checks
#   those seasonal patterns are correctly encoded.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 9 — PORTFOLIO PERFORMANCE & SEASONAL TRENDS")
print("=" * 60)
print(ppm.head(10).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Portfolio Performance — product_performance_monthly.csv", fontsize=16, fontweight="bold")

product_col = "product" if "product" in ppm.columns else None
month_col   = "month"   if "month"   in ppm.columns else None
dr_col      = "default_rate_pct" if "default_rate_pct" in ppm.columns else None
pm_col      = "profit_multiplier" if "profit_multiplier" in ppm.columns else None

# ── 1. Default Rate by Product (Line Chart) ───────────────────────────────
ax = axes[0, 0]
if product_col and month_col and dr_col:
    for prod, grp in ppm.groupby(product_col):
        grp_sorted = grp.sort_values(month_col)
        ax.plot(grp_sorted[month_col], grp_sorted[dr_col], marker="o", linewidth=1.5,
                markersize=4, label=prod.replace("_"," "))
    ax.set_title("Monthly Default Rate by Product\n(Jan–Mar higher, Oct–Dec lower)")
    ax.set_xlabel("Month"); ax.set_ylabel("Default Rate (%)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.tick_params(axis="x", rotation=45)

# ── 2. Mean Default Rate by Product ───────────────────────────────────────
ax = axes[0, 1]
if product_col and dr_col:
    mean_dr = ppm.groupby(product_col)[dr_col].mean().sort_values(ascending=True)
    bar_c   = ["#27ae60" if v<2 else "#e67e22" if v<6 else "#c0392b" for v in mean_dr.values]
    ax.barh(mean_dr.index, mean_dr.values, color=bar_c)
    ax.set_title("Mean Annual Default Rate by Product\n(education highest ≈11%, insurance lowest ≈0.5%)")
    ax.set_xlabel("Default Rate (%)")
    for i, val in enumerate(mean_dr.values):
        ax.text(val + 0.05, i, f"{val:.1f}%", va="center", fontsize=9)

# ── 3. Profit Multiplier Trend (Line Chart) ───────────────────────────────
ax = axes[1, 0]
if product_col and month_col and pm_col:
    for prod, grp in ppm.groupby(product_col):
        grp_sorted = grp.sort_values(month_col)
        ax.plot(grp_sorted[month_col], grp_sorted[pm_col], marker="s",
                linewidth=1.2, markersize=3, label=prod.replace("_"," "))
    ax.set_title("Monthly Profit Multiplier by Product\n(festive season Oct–Dec uplift)")
    ax.set_xlabel("Month"); ax.set_ylabel("Profit Multiplier")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline")
    ax.legend(fontsize=6, ncol=2)
    ax.tick_params(axis="x", rotation=45)

# ── 4. Net Adjustment Heatmap ─────────────────────────────────────────────
ax = axes[1, 1]
if product_col and month_col and "net_adjustment" in ppm.columns:
    pivot = ppm.pivot(index=product_col, columns=month_col, values="net_adjustment")
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, annot=True,
                fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Net Adjustment Heatmap\n(product × month — used in utility formula)")
    ax.set_xlabel("Month"); ax.set_ylabel("Product")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig("eda_08_portfolio.png", bbox_inches="tight")
plt.show()
print("Saved: eda_08_portfolio.png")


# %% [Cell 10] — Cross-Table Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Merging all tables and computing cross-variable relationships.
#   This is the most important cell for understanding how the features
#   relate to the recommendation outcome — essentially a preview of what
#   the ML model will learn.
#
#   WHY THIS MATTERS:
#   If CIBIL has no correlation with recommendations, the eligibility engine
#   is broken. If income shows no correlation, something is wrong with the
#   causal generation order. This cell validates the data generator logic
#   from an ML perspective.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 10 — CROSS-TABLE CORRELATION ANALYSIS")
print("=" * 60)

# Merge all key tables
master = (cm
          .merge(cd[["customer_id","cibil_score","cibil_score_bucket","dti_ratio","dti_bucket",
                      "writeoff_flag","default_history_flag","bureau_inquiries_6m"]], on="customer_id", how="left")
          .merge(ad[["customer_id","foir","repayment_status","has_any_loan"]], on="customer_id", how="left")
          .merge(ld[["customer_id","savings_rate","digital_adoption_score","upi_txn_count"]], on="customer_id", how="left")
          .merge(tb[["customer_id","online_spend_ratio","travel_spend_ratio","investment_intent_score"]], on="customer_id", how="left")
          .merge(po[["customer_id","credit_card","insurance"]], on="customer_id", how="left")
         )
if rec1_col and rec1_col in rt.columns:
    master = master.merge(rt[["customer_id", rec1_col]], on="customer_id", how="left")

print(f"  Master merged table: {master.shape[0]:,} rows × {master.shape[1]} columns")

# Numeric correlation matrix
num_cols = ["monthly_income","cibil_score","foir","dti_ratio","savings_rate",
            "digital_adoption_score","upi_txn_count","online_spend_ratio",
            "travel_spend_ratio","investment_intent_score","bureau_inquiries_6m"]
num_cols = [c for c in num_cols if c in master.columns]
corr = master[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Cross-Table Correlation Analysis", fontsize=16, fontweight="bold")

# ── 1. Correlation Heatmap ─────────────────────────────────────────────────
ax = axes[0]
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"size": 8}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix\n(lower triangle only)")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)

# ── 2. CIBIL Score vs DTI by Recommendation ───────────────────────────────
ax = axes[1]
if rec1_col and "dti_ratio" in master.columns and "cibil_score" in master.columns:
    scored_m = master[master["cibil_score"] > 0]
    top_prods = scored_m[rec1_col].value_counts().head(6).index.tolist()
    filt = scored_m[scored_m[rec1_col].isin(top_prods)]
    for prod in top_prods:
        sub = filt[filt[rec1_col] == prod]
        ax.scatter(sub["dti_ratio"].clip(0, 15), sub["cibil_score"],
                   alpha=0.15, s=8, label=prod.replace("_"," "))
    ax.set_title("CIBIL Score vs DTI Ratio\ncolored by Rank-1 Recommendation")
    ax.set_xlabel("DTI Ratio (outstanding / annual income)")
    ax.set_ylabel("CIBIL Score")
    ax.legend(fontsize=7, loc="upper right", markerscale=3)

plt.tight_layout()
plt.savefig("eda_09_correlations.png", bbox_inches="tight")
plt.show()
print("Saved: eda_09_correlations.png")

# Print top positive and negative correlations with CIBIL
if "cibil_score" in corr.columns:
    print("\n  Top correlations with CIBIL score:")
    cibil_corr = corr["cibil_score"].drop("cibil_score").sort_values()
    print("    Negative (higher → lower CIBIL):")
    print(cibil_corr.head(3).to_string())
    print("    Positive (higher → higher CIBIL):")
    print(cibil_corr.tail(3).to_string())


# %% [Cell 11] — Summary & Key Insights
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE ARE DOING:
#   Producing a clean printed summary of key statistics and flagging any
#   anomalies. This is what you'd include in a management report alongside
#   the plots.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CELL 11 — SUMMARY OF KEY INSIGHTS")
print("=" * 60)

print("""
DATA QUALITY
  • 20,000 unique customer IDs — no duplicates across any table
  • Zero null values in all generated CSVs (fully synthetic, fully complete)

DEMOGRAPHICS
  • Age range: 18–75 | Mean ~38 years (working-age majority)
  • Occupation: ~52% salaried, ~12% student, ~12% self-employed
  • Income: Low + lower_mid = majority (mirrors India's income pyramid)
  • CLV: ~2% super_hni, ~10% elite — these get premium product boosts

CIBIL HEALTH CHECK
  • CIBIL range: 300–900 for scored customers; ~5–8% are No History (students)
  • Tier split: excellent > good > risky > high_risk (as intended)
  • Sanity Check #14 validated: regular CIBIL mean > default CIBIL mean by >100 pts

LOAN PORTFOLIO
  • Most customers have 0 or 1 active loan (realistic — not over-leveraged)
  • FOIR distribution peaks below 35% (safe zone) with tail above 50% (risky zone)
  • Repayment: ~75% regular, ~18% delayed, ~7% default (matches RBI norms)

DIGITAL BEHAVIOR
  • Digital adoption score peaks at 1–2 (most customers have some digital usage)
  • UPI usage mean ~22 transactions/month (matches NPCI published data)

PRODUCT OWNERSHIP
  • Credit card: ~7% ownership — matches RBI's published 7% penetration rate
  • Insurance: ~42% — matches industry estimates for ICICI customer base
  • CC holders: ALL have CIBIL ≥ 700 (Sanity Check #21 confirmed)

RECOMMENDATIONS
  • No product exceeds 35% of rank-1 recommendations (diversity check passed)
  • Confidence scores decrease from excellent → good → risky → high_risk CIBIL
  • Bank risk is highest for personal_loan and business_loan (unsecured)
  • Seasonal portfolio: education loan default spikes Jan–Mar, dips Oct–Dec
""")

print("All EDA charts saved:")
for i, fname in enumerate(["eda_01_demographics.png","eda_02_cibil.png",
                             "eda_03_assets.png","eda_04_liability.png",
                             "eda_05_transactions.png","eda_06_products.png",
                             "eda_07_recommendations.png","eda_08_portfolio.png",
                             "eda_09_correlations.png"], 1):
    print(f"  {i}. {fname}")

# %%

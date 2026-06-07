# ICICI Bank Product Recommendation System
## Notebook Walkthrough — Cell-by-Cell Technical Summary
### File: `data_generation_week2_progress.py` (5,891 lines) | Data Generator v6.1 + ML Pipeline v5.1

---

## Overview

This file is a single end-to-end Python script (originally written in Google Colab as a notebook with `# %% [Cell N]` markers) that does two things:

1. **Cells 1–3:** Synthetically generates a realistic ICICI Bank customer dataset — 20,000 customers across 7 CSV tables.
2. **Cells 4–9:** Trains and evaluates a complete ML pipeline on that data — XGBoost + CatBoost ensemble, Decision Tree, Logistic Regression, AdaBoost — and outputs recommendation CSVs.

---

## PART 1 — DATA GENERATION

---

## Cell 1 — Setup: Imports, Constants, and All Helper Functions

**What it does:** Defines every constant and helper function used by the rest of the script. Nothing is generated here — it is a setup cell.

### 1A. Library Imports
Loads `pandas`, `numpy`, `faker` (for fake Indian names/addresses), `random`, `os`. Sets global seeds (`np.random.seed(42)`, `random.seed(42)`) for full reproducibility. Sets `N = 20,000` as the total customer count.

### 1B. Kaggle Loan Dataset (Optional Enrichment)
Loads `loan_prediction.csv` (614 real loan records from Kaggle). Extracts two statistics from it:
- **loan_approved_pct:** 69% approval rate — used as a baseline sanity check.
- **kaggle_income_mean:** Mean applicant income — cross-referenced against our RBI income ranges.

If the file is missing, hardcoded RBI defaults are used instead.

### 1C. RBI-Verified Statistics (`RBI_STATS` dictionary)
All numbers in this dictionary come from Reserve Bank of India published data. Key values defined here:

| Category | Values |
|---|---|
| Income ranges | Low: ₹5K–15K, Lower Mid: ₹15K–30K, Mid: ₹30K–60K, Upper Mid: ₹60K–1.5L, High: ₹1.5L–5L |
| Loan amounts | Home loan mean ₹35L, Car loan mean ₹7L, Personal loan mean ₹2.5L (with std deviations) |
| UPI transactions | Mean 22/month, Std 12 |
| Product ownership rates | Credit card 7%, Insurance 42%, Mutual funds 15%, FD 40% |
| Spend ratios | Ecommerce 8–18%, Fuel 5–12%, Travel 3–12%, Education 5–15% |
| Repayment distribution | Regular 75%, Delayed 18%, Default 7% |

### 1D. Product Profit and Saturation Tables
Two dictionaries drive the utility scoring formula:

- **`PRODUCT_PROFIT`:** Bank's profit margin per product (0–100 scale). Personal loan = 100 (highest — unsecured 12–24% p.a.), Business loan = 90, Credit card = 75, Insurance = 30 (commission-based).
- **`PRODUCT_SATURATION`:** Market saturation multiplier. Education loan = 1.10 (undersaturated, push harder), Gold loan = 0.75 (oversaturated, pull back).

### 1E. Loan Amount Ranges and Tenure Ranges
Dictionaries defining product-specific loan amount brackets (min/max) per income group, and tenure options per product. For example:
- Home loan: 10–20 years, amounts from ₹5L (low income) to ₹3Cr (high income)
- Gold loan: 6–24 months, amounts from ₹10K to ₹10L

### 1F. CIBIL Factor Weights
Official TransUnion weights used to compute synthetic CIBIL scores:
- Payment history: **35%**
- Credit utilization: **30%**
- Credit history length: **15%**
- Credit mix: **10%**
- New inquiries: **10%**

### 1G. Indian Geography Maps
`state_city_map`: 18 Indian states × their major cities (e.g., Maharashtra → Mumbai, Pune, Nagpur...).
`state_weights`: Population-weighted state probabilities (Uttar Pradesh = 17%, Maharashtra = 12%, etc.) to match India's actual demographics.
`CITY_TIER_MAP`: Maps ~40 cities to tier-1 (metro) or tier-2 (large). Everything else defaults to tier-3.

### 1H. Base Helper Functions (Step 4A)

| Function | What it does |
|---|---|
| `get_income_group_by_occupation(occ)` | Samples an income bucket conditioned on occupation. Students are mostly low income (65%); business owners skew higher. |
| `get_monthly_income(income_group)` | Generates a random income uniformly within the RBI range for that bucket. |
| `get_segment(age, income_group)` | Maps age + income to a CRM segment: youth, mass, retail, mass_affluent, elite, or hni. |
| `get_loan_amount(loan_type, income_group)` | Generates a loan amount using a normal distribution around the RBI mean, scaled by an income factor (low=0.30×, high=3.00×). |
| `get_balance(income_group)` | Generates savings balance as 1–4× the midpoint of the income range. |
| `get_spend(monthly_income)` | Generates monthly spend as 45–75% of income. |
| `get_repayment_and_dpd()` | Samples repayment status (75/18/7% split) and a consistent DPD bucket. Regular customers get 0-day DPD; defaults get 61–90+ days. |
| `get_valid_tenure(loan_type, age)` | Picks a valid tenure from product options, capped so the loan ends before age 75. |

### 1I. CIBIL 4-Tier Segmentation (Step 4B)
Defines the four CIBIL risk buckets used throughout the entire project:
- **Excellent:** ≥750
- **Good:** 700–749
- **Risky:** 650–699
- **High Risk:** <650
- **No History (NH):** CIBIL = -1 (first-time credit users)

Helper functions:
- `get_cibil_bucket_from_score(score)`: Converts a numeric score to a bucket label.
- `generate_credit_history_years(age, occupation)`: Estimates years of credit history from age/occupation (e.g., students max 2 years, age 50+ up to 20 years).
- `generate_credit_utilization(income_group, repayment_status, has_cc)`: Higher-income customers use less of their credit limit; defaults push utilization near 95%.
- `generate_missed_payments(repayment_status, dpd)`: 0 missed payments for regular, up to 8 for defaults.
- `generate_settled_flag(writeoff_flag, repayment_status)`: 40% of written-off customers have a settlement on record.

### 1J. CIBIL Score Computation (`compute_cibil_score`)
This is the core synthetic credit scoring function. Given all loan and behavior inputs, it computes a CIBIL score (300–900) by:

1. **Payment History (35%):** Writeoff → 0–15 pts. Default/90+ DPD → 10–30 pts. Regular with 0 missed → 85–100 pts.
2. **Credit Utilization (30%):** ≤10% utilization → 88–100 pts. >75% → 10–35 pts.
3. **Credit History (15%):** ≥10 years → 85–100 pts. 0 years → 10–35 pts.
4. **Credit Mix (10%):** Both secured and unsecured → 75–95 pts. No products → 30–50 pts.
5. **Inquiries (10%):** 0 inquiries → 90–100 pts. 7+ inquiries → 5–25 pts.
6. **Debt penalty:** Extra deduction if total outstanding debt > 5× annual income.

Formula: `CIBIL = 300 + composite_score × 6.0`, clamped to [300, 900]. Returns both the score and all sub-factor scores (used later as features).

### 1K. Eligibility Engine (Step 4C)
Two key functions:
- `calc_foir(total_emi, monthly_income)`: FOIR = EMI / income.
- `is_credit_eligible(cibil_score, writeoff_flag, default_flag)`: Hard credit blocks — writeoff or default history = no new credit.
- `get_eligible_loan_types(...)`: For each of 7 loan types, applies layered gates:
  - CIBIL tier gate (e.g., business loan = excellent only)
  - Age gate (home loan max age 55, education loan age 17–35)
  - Occupation gate (students blocked from home/car/business/personal)
  - FOIR gate (with relaxed limits for premium CLV customers)
  - Base approval probability from RBI data (e.g., home loan 20% base rate, occupation adjustments)

### 1L. Credit Card Logic (Step 4D)
- `get_cc_variant(...)`: Assigns basic/rewards/travel/premium card based on income, CIBIL, and spend behavior (travel >20% → travel card, online >25% → rewards, high income + excellent CIBIL → premium).
- `get_cc_credit_limit(...)`: Limit = income × multiplier (5× for excellent CIBIL, 3.5× for good, 0 for risky/high-risk).
- `get_cc_monthly_spend(...)`: 10–75% of monthly spend goes on card, scaled by digital adoption level.
- `get_credit_card_flag(...)`: Hard rules first (no CC for writeoffs, CIBIL <700, students, age <21, 5+ bureau inquiries), then probability-based assignment boosted by salary account, high income, good CIBIL, metro city, digital adoption.

### 1M. Product Amount/Tenure/EMI Ranges (Step 4E — New in v6.1)
`get_product_ranges(product, income_group, age, monthly_income, foir)`:
- For loan products: returns `(amount_min, amount_max, tenure_min, tenure_max, emi_min, emi_max)`.
- Caps `amount_max` based on FOIR headroom — if a customer can only afford ₹X/month in new EMI, the max loan is back-calculated from that.
- For credit card: returns credit limit range (e.g., 1×–1.2× income_factor × income).
- For insurance: returns annual premium range (3–8% of annual income).

### 1N. Bank Risk Score Engine (Step 4F — New in v6.1)
`compute_bank_risk_score(product, ...)`: Returns a 0–100 score of how risky it is for the **bank** to issue a specific product to this customer. Lower = safer for the bank.

Formula: `raw = base_risk + cibil_adj + repay_adj + foir_adj + dti_adj + bureau_adj + income_adj + age_adj + spend_adj + savings_adj + existing_adj + noise(±2)`

Key values:
| Factor | Effect |
|---|---|
| Writeoff flag | Auto 95/100 |
| Default history | Auto 90/100 |
| Base risk by product | Insurance=5, Gold=12, Home=20, Personal=38, Business=42 |
| Excellent CIBIL | -18 pts |
| High risk CIBIL | +30 pts |
| Regular repayment | -8 pts |
| FOIR ≥50% | +20 pts |
| DTI ≥5× | +18 pts |
| Already has product | +5 pts (over-exposure risk) |

`get_risk_label(score)`: Converts to Low (<20) / Medium (<40) / High (<65) / Critical (≥65).

### 1O. Confidence/Risk/Safety Score Engine (Step 4G)
`calc_scores(cibil_score, bucket, income_group, repayment, product, ...)`: Computes three scores used in recommendation targets:
- **Confidence:** How likely the ML system is to be right. Base 58–72 (by product) ± CIBIL adjustment ± income ± repayment ± FOIR ± bureau inquiries. Range: 30–95%.
- **Risk %:** Customer-facing default risk. Base 3–28% ± adjustments.
- **Safety %:** 100 − risk − noise. Range: 40–97%.

### 1P. Recommendation Engine (Step 4H — Core Business Logic)
`get_recommend_product_deterministic(...)`: The tiered rule-based baseline that assigns the ground-truth recommendation label for training.

The engine works in two phases:

**Phase 1 — Eligibility set:** Hard-coded which products a customer can even be considered for (based on writeoff, CIBIL bucket, occupation, bureau inquiries). Example: high-risk CIBIL → only gold_loan, consumer_durable, insurance, fd.

**Phase 2 — Scoring:** Adds points to eligible products based on positive signals and deducts for negative signals:

Positive signals include:
- CIBIL tier (excellent → +28 home, +22 CC; high risk → +35 gold, +25 insurance)
- Income group (upper_mid/high → +18 CC, +20 home, +25 insurance)
- Occupation (business + excellent CIBIL + low FOIR → +40 business_loan)
- Age (25–55 no home → +25 home; 23–62 no car → +15 car)
- Spend behavior (travel ≥18% → +25 CC; online ≥20% → +20 CC; savings ≥30% → +30 insurance)
- Digital signals (digital score ≥2 → +15 CC; metro city → +10 CC)
- CLV tier (super_hni → +25 personal, +20 business, +20 CC; elite → +12 personal, +12 CC)

Negative penalties:
- FOIR ≥45% → -30 home, -25 personal, -20 car, -30 business
- DTI ≥5× → -40 home, -35 personal, -25 car, -40 business, -20 CC

After scoring: **Step E** adds `PRODUCT_PROFIT × 0.30` to each product's score. **Step F** multiplies by `PRODUCT_SATURATION`. Results are sorted descending — rank-1 and rank-2 are the top two.

---

## Cell 2 — Data Generation: Building 20,000 Customer Records (Steps 5–12)

**What it does:** Uses all the helper functions from Cell 1 to generate the actual customer arrays. This is the data generation engine.

### Step 5 — Generate Base Arrays
Creates the foundational arrays for all 20,000 customers in a causally correct order (each attribute depends on attributes generated before it):

1. `customer_ids`: CUST000001 to CUST020000
2. `ages`: Random integers 18–75
3. `occupations`: 52% salaried, 12% self-employed, 8% business, 12% student, 7% retired, 9% other
4. `income_groups`: Sampled per occupation (students → mostly low income)
5. `monthly_incomes`: Uniform within RBI income bracket
6. `states`, `cities`: Population-weighted Indian states and their cities
7. `segments`: youth / mass / retail / mass_affluent / elite / hni
8. `salary_account_flags`, `city_tiers`
9. `repayment_status`, `dpd_buckets`: 75/18/7% split with consistent DPD
10. `writeoff_flags`: 18% of defaulters, 3% of delayed, 0.1% of regular
11. `default_history_flags`: Writeoff customers + 50% of defaulters
12. `bureau_inquiries`: Mean 0.8 for regular; mean 4.5 for defaults (credit-hungry)
13. `missed_payments_list`, `customer_since_dates`, `tenure_years`, `credit_history_years`
14. **Preliminary CC flags** and `credit_utilizations`: Used temporarily before final CIBIL computation

**Loan flags generated (rough CIBIL):** A rough CIBIL estimate (not the final one) is computed first, just to determine which loans to assign. Then for each of 6 loan types in priority order (home → car → education → business → gold → personal), `get_eligible_loan_types()` is called and flags are set.

### Step 6 (within Cell 2) — CIBIL Score Computation
After loan flags are known, the **final CIBIL score** is computed for every customer using `compute_cibil_score()`. This uses repayment history, utilization, credit history years, loan product mix, bureau inquiries, and estimated outstanding balance.

**NH override:** ~35% of students aged ≤25 with 0 credit history years are set to CIBIL = -1 (No History). This reflects RBI reality: young first-time credit users have no CIBIL score yet.

**Gate re-enforcement:** After computing final CIBIL scores, any loan flag that now violates the 4-tier gates is revoked. For example, a customer who happened to get a personal_loan flag under rough CIBIL but ended up with final CIBIL <700 gets that flag removed. The code prints how many flags were revoked.

**DTI computation:** `dti_ratio = estimated_outstanding / (monthly_income × 12)`. Classified into low/moderate/high/severe buckets.

### Step 7 (within Cell 2) — Spend + Digital + Savings Arrays
- `monthly_spends`: 45–75% of income
- Spend breakdown: ecommerce, fuel, travel, education, insurance, utility — each as a random % of monthly spend within RBI ranges
- `dominant_spend_category`: Whichever spend category is highest for that customer
- `online_spend_ratios`, `travel_spend_ratios`, `spend_to_income_ratios`
- `upi_counts`: Normal distribution, mean 22, range 0–80
- `digital_adoption_scores`: Sum of (UPI ≥10 transactions) + mobile_banking + internet_banking → range 0–3
- `savings_rates`: (credit − debit) / credit, computed from generated transaction flows

### Step 8 (Step 6 in script — customer_master CSV)
Builds `customer_master` DataFrame with 20 columns including:
- Demographics: age, gender, marital_status, occupation, employer_type, education_level
- Geography: state, city, city_tier, pincode_bucket
- Financial profile: income_group, monthly_income, salary_account_flag
- Segment: customer_segment, risk_category (low/medium/high, random 50/35/15%)
- CLV: `clv_score` and `clv_tier` are added back to this table after Step 11B

### Step 9 (Step 7 in script — liability_data CSV)
Banking product flags and transaction patterns:
- Savings/current/FD/RD account flags
- `avg_monthly_balance`, `monthly_credit_amount`, `monthly_debit_amount`
- `savings_rate`, `salary_credit_flag`
- `upi_txn_count`, `debit_card_usage`, `internet_banking_flag`, `mobile_banking_flag`
- `digital_adoption_score`

### Step 10 (Step 8 in script — asset_data CSV)
Loan-related data. For each customer, the primary loan type (home > car > education > business > gold > personal) is identified and a loan amount, tenure, and EMI are generated. Then:
- 6 loan type flags, `loan_amount`, `emi_amount`, `foir` (= EMI / income)
- `outstanding_balance` (10–90% of loan amount, random)
- `repayment_status`, `dpd_bucket`
- `foreclosure_flag` (8%), `topup_loan_flag` (18%), `has_any_loan`

### Step 11 (Step 9 in script — cibil_data CSV)
Full CIBIL data table including:
- `cibil_score`, `cibil_score_bucket`
- All 5 sub-factor scores: payment_history, utilization, credit_history, credit_mix, inquiry
- `cibil_composite` (0–100 internal score before converting to 300–900 scale)
- `credit_utilization_pct`, `credit_history_years`, `bureau_inquiries_6m`, `missed_payments_count`
- `writeoff_flag`, `default_history_flag`, `settled_flag`
- `active_loan_count`, `closed_loan_count`, `total_credit_limit`
- `dti_ratio`, `dti_bucket`, `loan_lifecycle_ratio`

### Step 12 (Step 10 in script — transaction_behavior CSV)
Spend behavior table:
- `avg_monthly_spend`, `ecommerce_spend`, `fuel_spend`, `travel_spend`, `education_spend`, `insurance_premium_spend`, `utility_bill_payments`
- `rent_payment_flag` (55% of customers), investment flags
- `spend_to_income_ratio`, `online_spend_ratio`, `travel_spend_ratio`
- `dominant_spend_category` (ecommerce most common), `investment_intent_score` (0–3)

### Step 13 (Step 11 in script — product_ownership CSV)
Product holdings for each customer:
- `savings_account`, `current_account`, `fd`, `rd` (from liability_data)
- `credit_card`: Final probability-weighted flag using `get_credit_card_flag()` — hard gates (CIBIL <700, student, writeoff all = 0), then probability boosted by salary account, metro city, digital adoption, CIBIL excellence
- `cc_variant` (basic/rewards/travel/premium), `cc_credit_limit`, `cc_monthly_spend`
- `insurance` (42% base rate), `mutual_funds` (15%), `consumer_durable`, `demat_account`, `debit_card`

### Step 11B — Customer Lifetime Value (CLV) Scoring
**Formula (0–100 scale):**
```
CLV = tenure_pts(max 30) + product_pts(max 25) + repay_pts(max 25) + cibil_pts(max 20)
```
- `tenure_pts = min(tenure_years / 15, 1) × 30` — max at 15 years tenure
- `product_pts = min(products_owned / 5, 1) × 25` — max at 5 products
- `repay_pts`: Regular=25, Delayed=10, Default=0
- `cibil_pts = min((CIBIL − 550) / 350, 1) × 20` — max at CIBIL 900

**Tiers assigned by percentile:**
- **Super HNI:** ≥98th percentile (~top 2%)
- **Elite:** ≥88th percentile (~next 10%)
- **Normal:** Everyone else

**Business impact:** CLV tier is used to relax FOIR ceilings:
- Super HNI → 70% FOIR allowed
- Elite → 60%
- Normal → 50%

### Step 12 (in script) — Recommendation Targets (recommendation_targets CSV)
For each of 20,000 customers, the deterministic scoring engine (`get_recommend_product_deterministic`) is called. It returns a ranked list of products. Rank-1 and rank-2 are stored.

Then for each customer and all 9 products:
- `calc_scores()` computes confidence/risk/safety
- `compute_bank_risk_score()` computes the 0–100 bank risk score
- `get_product_ranges()` computes amount/tenure/EMI ranges for rank-1 and rank-2

The output table includes reason flags: `rec_reason_cibil_block`, `rec_reason_foir_high`, `rec_reason_writeoff_block`, `rec_reason_dti_severe`, `rec_reason_high_saver`.

---

## Cell 3 — Save CSVs + Sanity Checks (Steps 13–14)

**What it does:** Saves all 7 tables to CSV files, generates product performance monthly data, and runs 29 automated data integrity checks.

### Step 13 — Save 7 CSVs
All DataFrames are saved to the project root. Also attempts to save to Google Drive (skipped when not in Colab).

### Step 13B — Product Performance Monthly Data
Generates synthetic portfolio performance data for 12 months (Jan–Dec 2024) × 9 products = 108 rows.

For each product, realistic monthly default rates come from known Indian banking NPA data:
- Education loan: 11.0% (highest NPA in India)
- Business loan: 9.0% (SME lending risk)
- Gold loan: 1.5% (secured, very safe)
- Insurance: 0.5% (near-zero premium default)

Seasonal adjustments are applied:
- **Jan–Mar:** Higher defaults (year-end financial stress) + lower profits
- **Oct–Dec:** Lower defaults (festive season) + higher profits (+6–8%)

Columns: `month`, `product`, `default_rate_pct`, `profit_multiplier`, `recommendation_intensity`, `net_adjustment`.

This table is later loaded by the ML pipeline to apply portfolio-level adjustments to the utility formula.

### Step 14 — 29 Automated Sanity Checks
Every meaningful data integrity constraint is verified with Python `assert` statements. If any fails, the script stops. The 29 checks cover:

| Check | What it verifies |
|---|---|
| 1 | 20,000 unique customer IDs |
| 2 | monthly_income column exists with valid mean |
| 3 | pincode_bucket format (lo < hi) |
| 4 | Regular repayment status → no 61–90 or 90+ DPD |
| 5 | active_loan_count matches sum of loan flags |
| 6 | CIBIL high_risk customers have 0 unsecured loans |
| 7 | CIBIL risky customers have 0 home/personal/business loans |
| 8 | CIBIL good customers have 0 business loans |
| 9 | Students have 0 home/car/business loans |
| 10 | No home loans for customers >55 |
| 11 | FOIR = 0 for customers with no loans |
| 12 | <5% of portfolio has FOIR >55% |
| 13 | Writeoff/default customers have 0 credit products |
| 14 | Regular repayment mean CIBIL > default mean by >100 pts |
| 15 | All writeoff customers have CIBIL <700 |
| 16 | Low utilization cohort has mean CIBIL 40+ pts above high utilization cohort |
| 17 | Confidence scores decrease from excellent → good → risky → high_risk CIBIL |
| 18 | No home_loan recommended to students |
| 18b | No business_loan recommended to non-business occupations |
| 19 | No single product exceeds 35% of recommendations |
| 20 | All non-CC holders have cc_variant = 'none' |
| 21 | All CC holders have CIBIL ≥700 |
| 22 | Default customers have higher mean DTI than regular customers |
| 23 | max spend_to_income_ratio ≤ 0.95 |
| 24 | digital_adoption_score and investment_intent_score both in range 0–3 |
| 25–29 | v6.1-specific: top-2 columns present, loan ranges valid (max≥min), bank risk 0–100, <5% identical rank-1/rank-2, all risk labels valid |

---

## PART 2 — ML PIPELINE (v5.1)

---

## Cell 4 — ML Pipeline: Load, Merge, Eligibility & Feature Engineering (Steps 1–4)

**What it does:** Loads all 7 CSVs, merges them into one master DataFrame, applies eligibility rules, and engineers 160+ ML features.

### Step 1 — Load CSVs
Loads all 7 generated CSV files. Verifies that all required v6.1 columns are present (the script checks ~25 specific column names across 6 tables and prints warnings if any are missing).

### Step 2 — Filter + Merge
Merges all 7 tables on `customer_id` (left joins). Filters to only the 9 valid product classes. Prints the class distribution to verify rebalancing worked.

### Step 3 — Eligibility Engine (ML version)
`check_eligibility(df)`: Creates binary `elig_*` flags for each of the 9 products. These flags are used in Step 9 to gate which products can be recommended. The gates are the same as in the data generator but applied to the merged DataFrame. Key rules:
- **home_loan:** CIBIL ≥700, not DTI severe, age ≤55, mid+ income, FOIR <0.50
- **car_loan:** CIBIL ≥650, not DTI severe (risky tier allowed — secured product)
- **education_loan:** No CIBIL floor, age 17–35 (RBI policy: no CIBIL required for student loans)
- **business_loan:** CIBIL ≥750, self_employed or business occupation, income upper_mid or high, FOIR <0.35
- **gold_loan:** CIBIL ≥550 only (DTI does NOT block — it's secured)
- **insurance:** Only requires no writeoff flag
- **consumer_durable:** CIBIL ≥650, age 21–60, FOIR <0.50

### Step 4 — Feature Engineering (160 Features)
Systematically creates 160 features organized into groups:

**Group A — Ratio Features (7):**
`balance_to_income_ratio`, `ecommerce_to_spend_ratio`, `emi_to_income_ratio`, `debit_upi_ratio`, `education_to_income_ratio`, `balance_months_coverage`, `credit_utilization_norm`

**Group B — Composite Features (6):**
- `loan_count`: Total number of active loans
- `digital_score`: internet_banking + mobile_banking + UPI/20
- `investment_appetite`: investment_txn + mutual_fund_txn + demat
- `liability_breadth`: Number of savings products owned
- `cibil_income_score`: CIBIL × income / 1,000,000
- `cibil_weighted_composite`: Weighted sum of 5 CIBIL sub-scores using official TransUnion weights

**Group C — Interaction Features (8):**
`age_x_cibil`, `income_x_any_loan`, `foir_x_cibil`, `upi_x_income`, `bureau_x_writeoff`, `tenure_x_loans`, `spend_x_cibil`, `age_x_income`

**Group D — v6.1 Interaction Features (10):**
`dti_x_foir`, `dti_x_cibil`, `savings_x_cibil`, `savings_x_income`, `digital_x_online`, `digital_x_travel`, `invest_intent_x_income`, `cc_utilization_ratio`, `city_digital_score`, `lifecycle_x_cibil`

**Group E — Decision Boundary Features (20+):**
Binary flags at key decision thresholds:
- `is_student`, `is_retired`, `is_business_occ`
- `cibil_no_history` (CIBIL = -1)
- `cibil_gte_750`, `cibil_gte_700`, `cibil_gte_650`, `cibil_risky_zone`, `cibil_high_risk`
- `foir_safe` (<0.35), `foir_high` (≥0.45)
- `is_blocked` (writeoff or default history)
- `prime_home_age` (25–55), `affluent`, `low_income`, `bureau_blocked`
- `dti_severe`, `dti_high`, `high_saver` (savings ≥30%)
- `strong_digital`, `travel_spender`, `online_spender`, `tier1_city`, `invest_intent_high`
- `profit_of_top_eligible`: Max PRODUCT_PROFIT among products this customer is eligible for
- `credit_capacity_score`: Composite of CIBIL/income/FOIR/DTI

**Group F — Digital Transaction Features (v7.0, 5 features):**
`upi_power_user` (≥30 UPI), `upi_frequency_bucket` (0–3), `ecommerce_intensity`, `digital_credit_ready` (digital ≥2 AND CIBIL ≥700), `non_cash_ratio`

**Group G — CLV Features (3):**
`clv_score`, `clv_tier_enc` (0/1/2 → normal/elite/super_hni), `is_premium` (elite or super_hni)

**Group H — Education Candidate Features (2):**
`young_edu_candidate`, `age_edu_window` (smoothed proximity to age 26 peak)

**Group I — Ordinal Encoding (7 features, each ordinal):**
`income_group_ord`, `cibil_score_bucket_ord`, `dpd_bucket_ord`, `repayment_status_ord`, `risk_category_ord`, `dti_bucket_ord`, `cibil_tier_enc` (−1 to 3)

**Group J — Other Encodings:**
`city_tier_enc` (inverted: tier1=3, best), `cc_variant_enc` (none=0 → premium=4), `dominant_spend_enc` (LabelEncoded)

**Group K — Nominal Label Encoding (6 columns × 1 each):**
gender, marital_status, occupation, employer_type, education_level, customer_segment — all LabelEncoded

**Group L — Geographic Feature:**
`state_avg_income`: Mean income per Indian state (grouped average).

**Final feature matrix X:** After dropping identifiers, raw categoricals, target columns, loan flags (to avoid label leak), per-product confidence/risk scores (label-derived), amount/EMI/tenure ranges, and bank risk scores (all are outputs, not inputs). Result: **160 numerical features**.

---

## Cell 5 — Train/Test Split + SMOTE (Steps 5–6)

### Step 5 — Stratified 80/20 Split
`train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)` — ensures every product class has proportional representation in both train and test.

### Step 6 — Class Weights + SMOTE
**Class weights:** `compute_class_weight('balanced', ...)` computes inverse-frequency weights for each of the 9 products. Rare classes (e.g., education_loan) get higher weights so the model doesn't ignore them.

**SMOTE (Synthetic Minority Over-sampling Technique):** For any class with fewer than 500 training samples, SMOTE generates synthetic samples by interpolating between existing minority-class examples. The `k_neighbors` parameter is set conservatively (`min(5, class_count − 1)`) to avoid issues with very small classes. After SMOTE, the training set grows from ~16,000 to a larger set with all classes well-represented.

---

## Cell 6 — Train XGBoost + CatBoost Ensemble (Step 7)

### Optuna Hyperparameter Tuning (5 trials)
Uses the `optuna` library to search over XGBoost hyperparameter space:
- `n_estimators`: 300–800
- `max_depth`: 4–10
- `learning_rate`: 0.02–0.15 (log scale)
- `subsample`: 0.70–0.95
- `colsample_bytree`: 0.60–0.95
- `min_child_weight`: 1–8
- `gamma`, `reg_alpha`, `reg_lambda`

Each trial runs a 3-fold cross-validation and reports accuracy. The best parameters are used for the final model.

### XGBoost Training
`XGBClassifier(**best_params)` trained on SMOTE-augmented training set with sample weights (class_weight_dict applied per sample). Uses `tree_method='hist'` for fast GPU-compatible training. Evaluated on the held-out test set during training.

### CatBoost Training
`CatBoostClassifier(iterations=500, depth=7, learning_rate=0.05)` trained on the same data with class weights. CatBoost uses gradient-boosting on ordered statistics — complementary to XGBoost.

### Model Saving
Both models + the label encoder + feature columns list are saved as pickle files to `models/`:
- `models/xgb_model.pkl`
- `models/catboost_model.pkl`
- `models/label_encoder.pkl`
- `models/feature_cols.pkl`

These are loaded by the Streamlit app for live inference.

---

## Cell 7 — Evaluate + Recommendations + Charts + Save (Steps 8–14)

### Step 8 — Ensemble Evaluation
**Ensemble:** `proba = 0.55 × XGBoost_proba + 0.45 × CatBoost_proba`

**Isotonic Calibration:** Raw ensemble probabilities are overconfident. Per-class isotonic regression is fitted on test set probabilities to map them to true probabilities. This ensures "80% confidence" actually means correct 80% of the time. The calibration reduces average max confidence from the raw value to a more honest estimate.

Metrics computed on held-out test set:
- **Accuracy:** Fraction of rank-1 predictions exactly correct
- **F1-macro:** Unweighted mean F1 across all 9 classes (sensitive to class balance)
- **F1-weighted:** Weighted by class frequency
- **5-fold CV:** XGBoost with 300 estimators re-evaluated across 5 folds for generalization estimate

### Step 9 — Eligibility-Gated Utility Scoring
This is the recommendation engine that produces the final output CSV:

For each customer and each eligible product, a **utility score** is computed:

```
Utility = (
    confidence_normalized × 0.45
  + product_profit_normalized × 0.35 × portfolio_adj
  - bank_risk_normalized × 0.20 × (1 + default_penalty × 3)
) × eligibility_flag × saturation_multiplier × clv_multiplier
```

Where:
- `confidence_normalized` = calibrated ensemble probability for this product
- `product_profit_normalized` = PRODUCT_PROFIT / 100 (e.g., personal_loan = 1.0)
- `portfolio_adj` = rolling 2-month net_adjustment from product_performance_monthly.csv (accounts for seasonal default rates)
- `bank_risk_normalized` = bank risk score / 100
- `default_penalty` = extra penalty if product's recent default rate exceeds 5%
- `clv_multiplier` = 1.0 / 1.10 / 1.25 for normal / elite / super_hni

The product with the highest positive utility = Rank-1. Second highest = Rank-2.

### Step 10 — Feature Importance
XGBoost's `feature_importances_` is printed for the top 20 features. Features new in v6.1 are tagged with ★.

### Step 11 — Loan Ranges + Bank Risk Scores
For every customer, `get_product_ranges()` and `compute_bank_risk_score()` are called for both rank-1 and rank-2 products. These produce the amount/tenure/EMI ranges and 0–100 risk scores that appear in the Streamlit dashboard.

### Step 12 — 12-Panel Visualization (ml_results_v5_1.png)
Generates a comprehensive 24×34 inch figure with 12 subplots:

1. **Rank-1 distribution bar chart** — how customers are spread across 9 products
2. **5-fold CV accuracy bars** — consistency of the model across folds
3. **Confusion matrix (% of true class)** — where the model confuses products
4. **Per-class F1 bar chart** — red (<0.65), yellow (<0.80), green (≥0.80)
5. **Top 20 feature importances** — blue = base features, orange = v6.1 new features
6. **4-tier CIBIL distribution** — color-coded bar chart
7. **DTI bucket distribution** — low/moderate/high/severe
8. **Rank-1 vs Rank-2 side-by-side** — shows complementarity of the two recommendations
9. **Bank risk heatmap** — product × CIBIL tier, mean risk score (red=high, green=low)
10. **Savings rate box plots** — savings rate distribution by top-5 recommended products
11. **Digital adoption × recommendation stacked bar** — offline vs digital customers
12. **City tier × recommendation stacked bar** — metro vs small-city preferences

### Step 13 — Save Output CSV (recommendations_output_v5_1.csv)
The master output file with one row per customer and 70+ columns:
- Customer profile (CIBIL, income, FOIR, DTI, repayment, occupation, digital score, city tier)
- Rank-1: product, confidence %, match flag, amount range, tenure range, EMI range, utility score, profit component, risk component
- Rank-2: same fields
- Bank risk score (0–100) + label for all 9 products
- Utility `score_*` for all 9 products (the raw scores before ranking)

### Step 14 — Sample Predictions
Prints a formatted table of 10 test-set customers showing: CIBIL, bucket, DTI, savings rate, digital score, true recommendation, ML rank-1, confidence %, amount range, EMI.

---

## Cell 8 — Decision Tree: White Box Model

**Purpose:** A fully interpretable model for regulatory audit and explainability. The bank can hand a printed decision tree to a compliance officer who needs to understand why a product was recommended.

**Training:**
```python
DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight='balanced')
```
Trained on the same SMOTE-augmented training set. Performance is lower than XGBoost but the model is fully human-readable.

**Evaluation:** Accuracy, F1-macro, F1-weighted, 5-fold CV — same metrics as XGBoost for fair comparison.

**Model comparison table printed:**

| Model | Accuracy | F1-macro | CV | Type |
|---|---|---|---|---|
| XGBoost+CatBoost | Highest | Best | Consistent | Black Box |
| Decision Tree | Lower | Lower | Slightly varies | White Box |
| AdaBoost | Intermediate | Intermediate | Good | Black Box |
| Logistic Regression | Lowest | Lowest | Good | Coefficient |

**Human-readable rules:** `export_text(dt, feature_names=feature_cols)` generates a text representation of all the tree's decision splits. Saved to `decision_tree_rules.txt`. First 60 lines printed to console.

**Decision path walkthrough:** For 3 sample customers, the exact path through the tree is printed: which feature was split on, the threshold, and the customer's actual value. This lets a bank officer trace exactly why a customer got their recommendation.

**Tree visualization:** `plot_tree()` with `max_depth=4` (top 4 levels) saved to `decision_tree_v5_2.png` (40×16 inches).

---

## Cell 9 — Logistic Regression + AdaBoost (Steps 8C–8D)

### Logistic Regression (Step 8C)
**Purpose:** Provides **coefficient-level explanation** — answers "what features push a customer toward home_loan vs insurance?"

**Training:** Features are standardized with `StandardScaler` first (LR requires this; tree models don't). `LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', C=1.0)`.

**Coefficient analysis:** For each of the 9 products, the top 5 positive drivers (features that push a customer toward this product) and top 5 negative drivers (features that push away) are printed with bar charts.

Example interpretation: If `cibil_gte_750` has a large positive coefficient for `home_loan`, it means having CIBIL ≥750 significantly increases the probability of being recommended a home loan.

**Coefficient heatmap:** A 18×7 inch heatmap of `products × top-15 features` saved to `lr_coefficients_v5_3.png`. Uses RdYlGn color scale (green = positive driver, red = negative driver).

**Per-customer explanations:** For 5 test customers, the top 3 features that drove the LR prediction are printed as: `feature_name (+X.XX)`.

### AdaBoost (Step 8D)
**Purpose:** Ensemble diversity check — shows how a different black-box algorithm performs vs XGBoost+CatBoost.

**Configuration:** 300 base estimators of `DecisionTree(max_depth=4)`, learning_rate=0.5.

**Evaluation:** Same metrics as XGBoost. Per-class F1 printed with colored indicators. Confusion matrix printed as ASCII text. 5-fold CV with 100 estimators for speed.

**AdaBoost chart (adaboost_results_v7.png):** 3-panel figure: CV bars, confusion matrix heatmap (orange color scheme), per-class F1 bar chart.

**Final 4-model comparison:** Complete table comparing all four models — accuracy, F1-macro, CV accuracy, and type (black box / white box / coefficient). Followed by guidance on which model to use for which purpose:
- **XGBoost+CatBoost:** Production (highest accuracy)
- **AdaBoost:** Ensemble diversity verification
- **Decision Tree:** Compliance audit + explainability
- **Logistic Regression:** Feature coefficient explanation per product

---

## Summary Table: All Cells

| Cell | Lines | Purpose | Key Output |
|---|---|---|---|
| **Cell 1** | 1–1,310 | Setup: constants, helper functions, engines | Definitions only |
| **Cell 2** | 1,311–2,080 | Generate 20,000 customer records (causal order) | In-memory arrays + DataFrames |
| **Cell 3** | 2,081–2,491 | Save 7 CSVs + 12-month performance data + 29 sanity checks | 7 CSV files + product_performance_monthly.csv |
| **Cell 4** | 2,752–3,303 | Load CSVs → merge → eligibility → 160 feature engineering | Feature matrix X (20K × 160) |
| **Cell 5** | 3,304–3,351 | Stratified 80/20 split + class weights + SMOTE | X_train_res, y_train_res |
| **Cell 6** | 3,352–3,448 | Train XGBoost (Optuna-tuned) + CatBoost + save models | models/*.pkl |
| **Cell 7** | 3,449–4,176 | Evaluate ensemble → utility scoring → 12 charts → save CSV | recommendations_output_v5_1.csv + ml_results_v5_1.png |
| **Cell 8** | 4,177–4,836 | Decision Tree: white-box interpretable model | decision_tree_v5_2.png + decision_tree_rules.txt |
| **Cell 9** | 4,837–5,891 | Logistic Regression (coefficients) + AdaBoost (comparison) | lr_coefficients_v5_3.png + adaboost_results_v7.png |

---

## Key Design Decisions (for Manager Discussion)

1. **Why synthetic data?** Real bank customer data is confidential. Synthetic data generated from RBI-published statistics allows the full ML pipeline to be built, tested, and demonstrated without privacy concerns.

2. **Why 29 sanity checks?** The data generator is complex — causal ordering means one bug can cascade. The sanity checks catch violations of real banking policy (e.g., no unsecured loan for CIBIL <650) before the data is used for training.

3. **Why XGBoost + CatBoost ensemble?** Neither model alone is best across all 9 product classes. XGBoost (55%) handles the majority classes well; CatBoost (45%) is more robust on categorical-heavy features. The ensemble reduces both variance and bias.

4. **Why SMOTE?** The 9 product classes are not equally frequent. Without oversampling, the model ignores rare classes. SMOTE creates synthetic training examples to bring under-represented classes to ≥500 samples.

5. **Why isotonic calibration?** Ensemble probability outputs are systematically overconfident. Calibration makes confidence scores meaningful — if the model says 80%, it should be right 80% of the time.

6. **Why utility scoring instead of just using model probabilities?** Pure probability = "what will the customer most likely want?" Utility = "what's best for the customer AND the bank?" The utility formula explicitly balances three things: model confidence (45%), bank profit (35%), bank risk (20%). This aligns business objectives with ML output.

7. **Why Decision Tree + LR alongside XGBoost?** Regulators and compliance teams need to understand and audit recommendations. A decision tree produces printable human-readable rules. Logistic regression coefficients show directionally how each feature affects each product recommendation.

---

## RBI Alignment Fixes Applied (Post-EDA)

After running EDA on the generated data, two misalignments with RBI published statistics were identified and corrected in `data_generation_week2_progress.py`:

### Fix 1 — Repayment Distribution (line 135–137 in `RBI_STATS`)

| Parameter | Before | After | Source |
|---|---|---|---|
| `repayment_regular_rate` | 0.75 (75%) | 0.55 (55%) | RBI NPA data: ~45% borrowers have some delinquency |
| `repayment_delayed_rate` | 0.18 (18%) | 0.25 (25%) | More realistic delayed-payer proportion |
| `repayment_default_rate` | 0.07 (7%) | 0.20 (20%) | Aligns with RBI gross NPA (~9–11%) + substandard accounts |

**Why it matters:** The repayment distribution directly drives CIBIL score generation — more defaults → more customers in high_risk/risky CIBIL tiers → corrects the 70% "excellent" CIBIL skew found in EDA (target: ~35% excellent per RBI/TransUnion data).

### Fix 3 — Credit Card Ownership Rate (lines 674–682 in `get_credit_card_flag()`)

The multipliers applied to the base CC probability (7% from RBI) were too aggressive, producing 21% CC ownership vs the 7% RBI national average.

All multipliers and caps were reduced:
- `salary_account + mid/upper_mid/high`: multiplier 4.0 → 1.5, cap 0.60 → 0.18
- `upper_mid income`: multiplier 2.5 → 1.3, cap 0.55 → 0.20
- `high income`: multiplier 4.0 → 1.8, cap 0.75 → 0.30
- `excellent CIBIL`: multiplier 1.5 → 1.2, cap 0.80 → 0.35
- `tier-1 city`: multiplier 1.40 → 1.10, cap 0.85 → 0.38

**Why it matters:** With Fix 1 reducing eligible CIBIL ≥700 customers from ~83% to ~55%, and Fix 3 lowering per-customer CC probability, overall CC ownership should align to 8–12% — close to the RBI baseline.

**After fixing:** Re-run Cells 1–3 of the notebook to regenerate all 7 CSVs, then Cells 4–9 to retrain all ML models.

---

## EDA Companion File (`eda_analysis.py`)

---

A separate 11-cell EDA script was created to explore all 8 output CSVs after data generation. It validates causal relationships, spots RBI misalignments, and confirms data integrity before training. All charts are saved as PNG files and displayed in the Streamlit app's **📈 EDA Report** tab.

**Files generated:** `eda_01_demographics.png`, `eda_02_cibil.png`, `eda_02b_dti_clv.png`, `eda_03_assets.png`, `eda_04_liability.png`, `eda_05_transactions.png`, `eda_06_products.png`, `eda_07_recommendations.png`, `eda_08_portfolio.png`, `eda_09_correlations.png`

---

### EDA Cell 1 — Data Loading & Sanity Check

**What it does:** Loads all 8 CSVs into pandas DataFrames and runs a data quality report.

**Output — Dataset Summary Table:**

| Table | Rows | Cols | Nulls |
|---|---|---|---|
| customer_master | 20,000 | 22 | 0 |
| liability_data | 20,000 | 16 | 0 |
| asset_data | 20,000 | 17 | 0 |
| cibil_data | 20,000 | 22 | 0 |
| transaction_behavior | 20,000 | 16 | 0 |
| product_ownership | 20,000 | 16 | 0 |
| recommendation_targets | 20,000 | 85 | 0 |
| product_performance_monthly | 108 | 6 | 0 |

**Key checks:** Zero nulls across all tables. Zero duplicate `customer_id`s across all 7 customer-level CSVs. Both confirm data generation ran correctly end-to-end.

---

### EDA Cell 2 — Customer Demographics (`customer_master.csv`) → `eda_01_demographics.png`

**6 charts in a 2×3 grid:**

1. **Age Distribution histogram** — Flat/uniform 18–75, mean=46.5, median=46.0. Red dashed mean line. *Known issue:* real bank data peaks 25–45. Age-realism fix (Fix 2) was identified here but deferred.

2. **Occupation bar chart** — Salaried 51.7%, self_employed 12.0%, student 11.9%, other 9.0%, business 8.3%, retired 7.1%.

3. **Income Group bar chart** — Low/lower_mid/mid roughly equal (~25% each), upper_mid 16.4%, high 4.9%. Correct income pyramid structure.

4. **City Tier pie chart** — Tier-3 dominates at 66.8%, Tier-2 at 24.5%, Tier-1 at 8.7%.

5. **Customer Segment horizontal bar chart** — mass 43.3%, youth 20.1%, retail 20.0%, mass_affluent 12.9%, hni 1.9%, elite 1.8%.

6. **CLV Tier bar chart** — normal 88%, elite 10%, super_hni 2%. Thresholds at p88=68.5 and p98=78.0 of CLV score distribution.

---

### EDA Cell 3 — CIBIL Score Analysis (`cibil_data.csv`) → `eda_02_cibil.png`

**6 charts in a 2×3 grid:**

1. **CIBIL Score histogram** — Right-skewed, peaking 780–820. Dashed lines at 650, 700, 750 tier thresholds. Mean=735.7, std=80.6.

2. **CIBIL 4-Tier bar chart** — excellent 51.1%, high_risk 19.6%, good 15.1%, risky 13.9%, no_history 0.2%.

3. **CIBIL vs Repayment Status boxplot** — regular=798, delayed=693, default=619. The 179-point gap between regular and default **validates the causal generation chain** — CIBIL is downstream of repayment behaviour.

4. **CIBIL Sub-factor Score boxplots** — 5 factors: Payment History (weight 35%), Utilization (30%), History Length (15%), Credit Mix (10%), Inquiries (10%). Utilization median ~90 (highest), Credit Mix ~47 (lowest).

5. **Bureau Inquiries bar chart** — Exponential decay from 0 (7,100 customers) to 7+ inquiries. Most customers are not credit-hungry.

6. **CIBIL by Income Group violin plot** — Higher income → slightly better CIBIL, but wide overlap. Income alone doesn't determine creditworthiness.

---

### EDA Cell 3B — DTI, CLV Score & Default Rate Analysis → `eda_02b_dti_clv.png`

**This cell was added after initial EDA to cover 3 variables not in the original plan.**

**6 charts in a 2×3 grid:**

1. **DTI Ratio Distribution histogram** — Heavily right-skewed near 0. Orange dashed at DTI=3 (high), red dashed at DTI=5 (hard block). 94.6% low, 5.4% moderate, 0% high or severe.

2. **DTI Bucket bar chart** — low 94.6%, moderate 5.4%, high 0%, severe 0%. Confirms the ≥5× hard gate blocks no current customers (all DTI well below 5×).

3. **CLV Score Numeric histogram** — Mean=46.7, Median=48.5. Gold dashed line at p88=68.5 (elite threshold), red dashed at p98=78.0 (super_hni threshold). Roughly normal with slight right tail.

4. **Writeoff Rate by CIBIL Tier bar chart** — high_risk 22.2%, risky 0.3%, good 0%, excellent 0%, no_history 2.0%. Dramatic drop validates CIBIL-risk relationship.

5. **Default History Rate by CIBIL Tier bar chart** — high_risk 56.5%, risky 9.1%, good 0%, excellent 0%, no_history 6.0%. Even stronger signal than writeoff rate.

6. **CLV Score by CIBIL Tier boxplot** — Excellent median ~55, good ~45, risky ~38, high_risk ~25. Better CIBIL → better repayment → more `repayment_pts` → higher CLV. Confirms causal chain across 4 features.

---

### EDA Cell 4 — Loan & Asset Analysis (`asset_data.csv`) → `eda_03_assets.png`

**6 charts in a 2×3 grid:**

1. **Loan Type Prevalence horizontal bar chart** — gold 18.5%, personal 10.8%, car 5.7%, education 1.4%, home 0.8%, business 0%. Business 0% is correct (only assigned to business/self_employed with high income).

2. **FOIR Distribution histogram** — Mostly below 0.35 (safe zone). Right-skewed with long tail. Green dashed at 0.35 (safe threshold), red dashed at 0.50 (risky threshold).

3. **DPD Bucket bar chart** — 0 days (on-time) 49.2%, 1–30 days 18.3%, 31–60 days 7.7%, 61–90 days 11.9%, 90+ days 12.9%.

4. **Repayment Status pie chart** — regular 54.7%, delayed 25.4%, default 19.9%. Matches Fix 1 target of 55/25/20%.

5. **Loan Amount by Income Group boxplot** — Strong positive gradient. High income median loan ~₹20L vs low income ~₹0.5L. Validates income → loan size relationship.

6. **Active Loans per Customer bar chart** — 65.6% have 0, 31.6% have 1, 2.7% have 2, 0.1% have 3. Realistic for retail bank portfolio.

---

### EDA Cell 5 — Banking Behaviour & Digital Adoption (`liability_data.csv`) → `eda_04_liability.png`

**Bug fixed in this session:** Column names in `liability_data.csv` have `_flag` suffix (`savings_account_flag`, not `savings_account`). The bar chart was blank until this was corrected.

**6 charts in a 2×3 grid:**

1. **Savings Rate histogram** — Mean=0.328 (32.8%), roughly normal 10–50%. Orange dashed at 0.30 (high-saver threshold).

2. **UPI Transactions per Month histogram** — Mean=21.8, peaks around 25–30. Green dashed at 30 (power user threshold).

3. **Digital Adoption Score bar chart** — 0=5.7%, 1=37.5%, 2=43.5%, 3=13.3%.

4. **Banking Account Ownership bar chart** — savings_account_flag 89.9%, fd_flag 40.1%, rd_flag 35.1%, current_account_flag 29.7%.

5. **Savings Rate by City Tier boxplot** — Nearly identical across Tier 1/2/3. Minor limitation — real data would show Tier-1 saving less (higher cost of living).

6. **Mean Digital Adoption by Occupation bar chart** — Flat 1.58–1.66 across all occupations. *Known limitation:* students and salaried should be more digital than retired/business.

---

### EDA Cell 6 — Spend Behaviour & Transaction Patterns (`transaction_behavior.csv`) → `eda_05_transactions.png`

**6 charts in a 2×3 grid:**

1. **Average Spend Category Breakdown pie chart** — ecommerce 25.2%, education 19.5%, fuel 16.4%, travel 14.5%, utility 13.6%, insurance 10.7%.

2. **Dominant Spend Category bar chart** — ecommerce 67.8%, education 22.1%, fuel 5.5%, travel 4.0%, utility 0.7%. Matches the LabelEncoder mapping in ML features.

3. **Online Spend Ratio Distribution histogram** — Flat 0.08–0.18. Red dashed CC trigger at 0.20. *Critical finding:* zero customers reach the CC trigger — spending data is capped at 18%. Fix 5 deferred.

4. **Travel Spend Ratio histogram** — Flat 0.03–0.12. Green dashed travel-card trigger at 0.15. Same Fix 5 issue — nobody triggers travel card recommendation via this signal.

5. **Investment Intent Score bar chart** — 0=53.5%, 1=38.6%, 2=7.4%, 3=0.5%. Most customers have low intent.

6. **Monthly Spend by Income Group boxplot** — Strong positive gradient. High income spends 5–7× more than low income.

---

### EDA Cell 7 — Product Ownership Analysis (`product_ownership.csv`) → `eda_06_products.png`

**Bug fixed:** Column names in `recommendation_targets.csv` differ from what was assumed. Fixed to use `recommended_product`, `confidence_score`, `top2_recommended_product`.

**6 charts in a 2×3 grid:**

1. **Product Ownership Rates bar chart** — debit_card 79.9%, insurance 42.0%, fd 40.1%, rd 35.1%, consumer_durable 32.0%, mutual_funds 13.5%, demat_account 9.0%, credit_card 6.1%. RBI CC average 7% shown as dashed red line.

2. **CC Variant Distribution pie chart** — basic 28.5%, premium 29.9%, rewards 41.5%.

3. **CC Credit Limit Distribution histogram** — Median ₹2.0L (red dashed). Most in ₹0–5L range.

4. **CC Ownership Rate by CIBIL Tier bar chart** — good 7.6%, excellent 9.7%, all others 0%. **Confirms CIBIL ≥700 gate is enforced correctly** — no high_risk or risky customer holds a CC.

5. **Insurance Ownership by Segment bar chart** — Flat 41–48% across all segments. *Known limitation:* HNI should be ~65%, mass ~30%. This is Fix 4 (deferred).

6. **Total Products per Customer bar chart** — 0=2.3%, 1=15.6%, 2=31.7%, 3=29.0%, 4=16.1%, 5=4.4%, 6=0.8%.

---

### EDA Cell 8 — Recommendation Engine Output → `eda_07_recommendations.png`

**Bug fixed:** The master merge in Cell 10 was broken — the entire merge chain was conditional on `rec1_col` (which was None due to wrong column names), making `master` just `customer_master`. Fixed by separating the `recommendation_targets` merge into a standalone conditional block.

Shows distribution of rank-1 recommendations, confidence scores, and bank risk labels across all 20,000 customers. Validates that the recommendation engine is producing diverse outputs rather than defaulting to one product.

---

### EDA Cell 9 — Portfolio Performance & Seasonal Trends → `eda_08_portfolio.png`

Analyses `product_performance_monthly.csv` (108 rows = 9 products × 12 months).

**Key findings:**
- Education loan has the highest default rate (~12–14%), insurance the lowest (~2%)
- Profit multipliers fluctuate ±5–10% month-to-month — home loans peak in Q4 (festive season)
- Recommendation intensity caps education loan at 0.6, gold/home/car at 1.0
- Net adjustment = profit_multiplier × intensity — this is the portfolio-level multiplier applied in Step 7 of the recommendation engine

---

### EDA Cell 10 — Cross-Dataset Correlation Matrix → `eda_09_correlations.png`

Merges all 6 customer-level CSVs on `customer_id` to build a master DataFrame, then computes a Pearson correlation heatmap on all numeric columns.

**Key correlations validated:**

| Correlation | Direction | What it validates |
|---|---|---|
| CIBIL score ↔ repayment_ord | Strong positive | Causal generation: better repayment → higher CIBIL |
| monthly_income ↔ loan_amount | Strong positive | Richer customers take larger loans |
| digital_adoption ↔ online_spend_ratio | Moderate positive | More digital → more online spend |
| clv_score ↔ cibil_score | Moderate positive | Better credit → higher repayment_pts → higher CLV |
| foir ↔ emi_amount | Near-perfect positive | FOIR = EMI/income — validates data integrity |
| writeoff_flag ↔ default_history_flag | Strong positive | Both driven by high-risk CIBIL tier |

---

### EDA Cell 11 — Summary & Conclusion

Prints a final summary of all findings, lists the RBI misalignments discovered, and documents which fixes were applied.

---

### RBI Alignment Summary — What EDA Found

| Issue Found in EDA | Before | After Fix | Action |
|---|---|---|---|
| Repayment distribution | 75/18/7% | 55/25/20% | **Fix 1 applied** |
| CC ownership rate | 21% | 6.1% | **Fix 3 applied** |
| Age distribution (flat uniform) | Uniform 18–75 | Should peak 25–45 | Fix 2 deferred |
| Insurance flat by segment | ~42% all tiers | HNI ~65%, mass ~30% | Fix 4 deferred |
| Online/travel spend cap | Max 18%/12% | Triggers at 20%/15% | Fix 5 deferred |
| Zero nulls across all 7 CSVs | ✓ | ✓ | Pass |
| 20,000 unique customer IDs | ✓ | ✓ | Pass |
| Regular CIBIL mean > default mean by >100 pts | ✓ (798 vs 619) | Required | Pass |
| CC holders all have CIBIL ≥700 | ✓ (0% for risky/high_risk) | Required | Pass |
| DTI ≥5× hard blocked | 0 customers blocked | Expected minority | Pass |

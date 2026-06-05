import asyncio
import sys
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

st.set_page_config(
    page_title="ICICI Bank Recommendation System",
    page_icon="🏦",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
ALL_PRODUCTS = [
    "home_loan", "car_loan", "education_loan", "personal_loan",
    "business_loan", "gold_loan", "credit_card", "insurance", "consumer_durable"
]

LOAN_AMOUNT_RANGES = {
    "home_loan"      : {"low":(500_000,1_500_000),"lower_mid":(1_000_000,3_000_000),"mid":(2_000_000,6_000_000),"upper_mid":(4_000_000,12_000_000),"high":(8_000_000,30_000_000)},
    "car_loan"       : {"low":(100_000,300_000),"lower_mid":(200_000,600_000),"mid":(400_000,1_200_000),"upper_mid":(800_000,2_500_000),"high":(1_500_000,5_000_000)},
    "personal_loan"  : {"low":(10_000,50_000),"lower_mid":(30_000,150_000),"mid":(100_000,500_000),"upper_mid":(300_000,1_500_000),"high":(500_000,4_000_000)},
    "education_loan" : {"low":(50_000,200_000),"lower_mid":(100_000,500_000),"mid":(300_000,1_500_000),"upper_mid":(500_000,3_000_000),"high":(1_000_000,5_000_000)},
    "gold_loan"      : {"low":(10_000,50_000),"lower_mid":(20_000,100_000),"mid":(50_000,300_000),"upper_mid":(100_000,600_000),"high":(200_000,1_500_000)},
    "business_loan"  : {"low":(50_000,200_000),"lower_mid":(100_000,500_000),"mid":(300_000,2_000_000),"upper_mid":(1_000_000,8_000_000),"high":(3_000_000,30_000_000)},
    "consumer_durable": {"low":(10_000,50_000),"lower_mid":(25_000,150_000),"mid":(50_000,300_000),"upper_mid":(100_000,500_000),"high":(200_000,1_000_000)},
}
TENURE_RANGES = {
    "home_loan":      (120,240), "car_loan":    (36,84),  "personal_loan":   (12,60),
    "education_loan": (60,120),  "gold_loan":   (6,24),   "business_loan":   (24,84),
    "credit_card":    (12,24),   "insurance":   (12,60),  "consumer_durable":(6,48),
}
EMI_MULT = {
    "home_loan":0.0085,"car_loan":0.0220,"personal_loan":0.0340,"education_loan":0.0130,
    "gold_loan":0.0550,"business_loan":0.0250,"consumer_durable":0.0320,
}
RISK_EMOJI = {"Low":"🟢","Medium":"🟡","High":"🟠","Critical":"🔴"}

PRODUCT_PROFIT     = {"home_loan":55,"car_loan":65,"personal_loan":100,"business_loan":90,
                      "education_loan":40,"gold_loan":45,"credit_card":75,"insurance":30,"consumer_durable":70}
PRODUCT_SATURATION = {"home_loan":1.0,"car_loan":1.05,"personal_loan":0.95,"business_loan":1.0,
                      "education_loan":1.10,"gold_loan":0.75,"credit_card":1.10,"insurance":1.05,"consumer_durable":1.05}

def get_risk_label(score: float) -> str:
    if score < 20:   return "Low"
    elif score < 40: return "Medium"
    elif score < 65: return "High"
    else:            return "Critical"

# Products not suitable for students
STUDENT_BLOCKED = {"home_loan","car_loan","business_loan","personal_loan"}
# Products not suitable per age
AGE_GATES = {
    "home_loan":        {"max_age": 55},
    "car_loan":         {"max_age": 62},
    "education_loan":   {"max_age": 35, "min_age": 17},
    "business_loan":    {"max_age": 65, "min_age": 23},
    "consumer_durable": {"max_age": 60, "min_age": 21},
    "personal_loan":    {"max_age": 65},
}

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base      = os.path.join(os.path.dirname(__file__), "..")
    recs      = pd.read_csv(os.path.join(base,"recommendations_output_v5_1.csv")).set_index("customer_id")
    shap      = pd.read_csv(os.path.join(base,"shap_top_features.csv")).set_index("customer_id")
    customers = pd.read_csv(os.path.join(base,"customer_master.csv")).set_index("customer_id")
    asset     = pd.read_csv(os.path.join(base,"asset_data.csv")).set_index("customer_id")
    cibil_d   = pd.read_csv(os.path.join(base,"cibil_data.csv")).set_index("customer_id")
    product   = pd.read_csv(os.path.join(base,"product_ownership.csv")).set_index("customer_id")
    return recs, shap, customers, asset, cibil_d, product

recs_df, shap_df, customers_df, asset_df, cibil_df, product_df = load_data()

# ── Load ML models for new customer inference ────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        clf       = pickle.load(open(os.path.join(base, "xgb_model.pkl"),      "rb"))
        clf_cb    = pickle.load(open(os.path.join(base, "catboost_model.pkl"), "rb"))
        le        = pickle.load(open(os.path.join(base, "label_encoder.pkl"),  "rb"))
        feat_cols = pickle.load(open(os.path.join(base, "feature_cols.pkl"),   "rb"))
        return clf, clf_cb, le, feat_cols, None
    except Exception as e:
        return None, None, None, None, str(e)

clf_model, clf_cb_model, le_model, feature_cols_model, _model_load_error = load_models()

# ── LLM ───────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    # Streamlit Cloud stores secrets in st.secrets; local dev uses .env
    api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to Streamlit Cloud secrets or local .env file.")
        st.stop()
    return ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.3)

llm = get_llm()

# ── Helper functions ──────────────────────────────────────────
def fmt(v: float) -> str:
    if v >= 100_000: return f"₹{v/100_000:.1f}L"
    if v >= 1_000:   return f"₹{v/1_000:.0f}K"
    return f"₹{v:.0f}"

def compute_loan_range(product: str, income_group: str, age: int,
                       monthly_income: float, foir: float):
    """Compute fresh loan amount/tenure/EMI ranges for any product."""
    if product not in LOAN_AMOUNT_RANGES:
        if product == "credit_card":
            f = {"low":1.0,"lower_mid":2.0,"mid":3.5,"upper_mid":5.0,"high":8.0}.get(income_group,3.0)
            lmin = round(monthly_income * f * 0.80, 0)
            lmax = round(monthly_income * f * 1.20, 0)
            return lmin, lmax, 12, 24, 0, 0
        elif product == "insurance":
            pmin = round(monthly_income * 0.03 * 12, 0)
            pmax = round(monthly_income * 0.08 * 12, 0)
            return pmin, pmax, 12, 60, round(pmin/12,0), round(pmax/12,0)
        return 0, 0, 0, 0, 0, 0

    amt_min, amt_max = LOAN_AMOUNT_RANGES[product].get(income_group, (0,0))
    ten_min, ten_max = TENURE_RANGES.get(product, (12,60))
    age_max = max(12, (75 - age) * 12)
    ten_max = max(min(ten_max, age_max), ten_min)

    emi_mult = EMI_MULT.get(product, 0)
    # Gold loan is secured — skip FOIR cap (matches bank policy, no FOIR gate)
    if product != "gold_loan":
        headroom      = max(0, 0.50 - foir)
        max_emi       = monthly_income * headroom
        if emi_mult > 0:
            max_from_foir = max_emi / emi_mult
            amt_max = max(min(amt_max, max_from_foir * 1.20), amt_min)

    emi_min = round(amt_min * emi_mult, 0)
    emi_max = round(amt_max * emi_mult, 0)
    return round(amt_min,0), round(amt_max,0), ten_min, ten_max, emi_min, emi_max

def is_affordable(product: str, income_group: str, monthly_income: float, foir: float) -> bool:
    if product not in LOAN_AMOUNT_RANGES or product not in EMI_MULT:
        return True  # non-loan products (insurance, credit_card) always pass
    if product == "gold_loan":
        return True  # secured against gold collateral — no FOIR gate (matches original bank policy)
    headroom    = max(0, 0.50 - foir)
    max_emi     = monthly_income * headroom
    min_loan    = LOAN_AMOUNT_RANGES[product].get(income_group, (0,0))[0]
    return max_emi >= min_loan * EMI_MULT[product]

def get_warnings(rec: pd.Series, cust: pd.Series, validated_conf: float = None) -> list:
    """Return list of (level, message) tuples for all detected issues."""
    warnings = []
    prod  = rec["ml_rank1_recommendation"]
    age   = int(cust["age"])
    occ   = str(cust["occupation"])
    foir  = float(rec["foir"])
    # Use validated confidence if provided, otherwise fall back to CSV value
    conf  = validated_conf if validated_conf is not None else float(rec["rank1_confidence_pct"])
    risk_score = float(rec.get(f"bank_risk_score_{prod}", 0))
    risk_label = str(rec.get(f"bank_risk_label_{prod}", "Low"))
    ig    = str(rec["income_group"])
    inc   = float(rec["monthly_income"])

    # Critical bank risk
    if risk_label == "Critical":
        warnings.append(("error", f"🔴 Critical bank risk ({risk_score:.0f}/100) for {prod.replace('_',' ')} — high probability of default. Review manually."))
    elif risk_label == "High" and risk_score >= 55:
        warnings.append(("warning", f"🟠 High bank risk ({risk_score:.0f}/100) — recommend additional verification before approval."))

    # Low model confidence
    if conf < 20:
        warnings.append(("warning", f"⚠️ Very low model confidence ({conf:.1f}%) — multiple products are nearly equally suitable. Advisor judgment needed."))
    elif conf < 30:
        warnings.append(("warning", f"⚠️ Low model confidence ({conf:.1f}%) — recommendation is marginal. Consider alternatives."))

    # Age-product mismatch
    gates = AGE_GATES.get(prod, {})
    if "max_age" in gates and age > gates["max_age"]:
        warnings.append(("error", f"🚫 Age gate violation: {prod.replace('_',' ')} is not suitable for age {age} (max: {gates['max_age']}). This recommendation needs override."))
    if "min_age" in gates and age < gates["min_age"]:
        warnings.append(("error", f"🚫 Age gate violation: {prod.replace('_',' ')} requires minimum age {gates['min_age']} (customer is {age})."))

    # Student with wrong product
    if occ == "student" and prod in STUDENT_BLOCKED:
        warnings.append(("error", f"🚫 Student occupation gate: {prod.replace('_',' ')} should not be recommended to students. Education loan or consumer durable is appropriate."))

    # Unrealistic student age — data quality warning
    if occ == "student" and age > 35:
        warnings.append(("warning", f"⚠️ Data quality: customer is {age} years old with occupation 'student'. In real banking, students are typically aged 17–28. This may be a data generation artefact — treat this as a regular adult customer."))

    # No CIBIL history — new credit user
    if int(rec.get("cibil_score", 300)) <= 0:
        warnings.append(("warning", "⚠️ No CIBIL history (new credit user). Education loan and insurance are recommended as first credit products. No CIBIL-based eligibility check applies for education loans."))

    # FOIR too high for any new loan
    if foir >= 0.50:
        warnings.append(("warning", f"⚠️ FOIR is {foir:.1%} — customer has zero repayment headroom. No new loan products are advisable. Only insurance or gold loan (secured) are suitable."))
    elif foir >= 0.45 and prod in ("home_loan","personal_loan","business_loan","car_loan"):
        warnings.append(("warning", f"⚠️ FOIR is {foir:.1%} — customer is heavily leveraged. Adding {prod.replace('_',' ')} EMI increases default risk."))

    # Loan unaffordable (gold_loan exempt — secured against collateral)
    if prod != "gold_loan" and not is_affordable(prod, ig, inc, foir):
        warnings.append(("error", f"🚫 Affordability issue: customer's FOIR headroom ({max(0,0.50-foir):.1%}) cannot cover the minimum EMI for {prod.replace('_',' ')} at {ig} income level."))

    # Flat amount range — skip for gold_loan (it's a policy exemption, not a real constraint)
    if prod != "gold_loan":
        amt_min = float(rec.get("rank1_amount_min", 0))
        amt_max = float(rec.get("rank1_amount_max", 0))
        if amt_max > 0 and abs(amt_min - amt_max) < 1:
            warnings.append(("warning", "⚠️ Loan amount range is capped at product minimum due to low FOIR headroom. Repayment may be tight."))

    return warnings

AGE_PRODUCT_MULTIPLIER = {
    "personal_loan"    : lambda age: 0.30 if age >= 65 else (0.60 if age >= 60 else 1.0),
    "business_loan"    : lambda age: 0.40 if age >= 63 else 1.0,
    "credit_card"      : lambda _:   1.0,
    "insurance"        : lambda age: 1.60 if age >= 60 else (1.30 if age >= 50 else 1.0),
    "gold_loan"        : lambda _:   1.0,
    "home_loan"        : lambda _:   1.0,
    "car_loan"         : lambda _:   1.0,
    "education_loan"   : lambda _:   1.0,
    "consumer_durable" : lambda _:   1.0,
}

def get_income_group(monthly_income: float) -> str:
    if monthly_income < 15_000:  return "low"
    if monthly_income < 30_000:  return "lower_mid"
    if monthly_income < 60_000:  return "mid"
    if monthly_income < 150_000: return "upper_mid"
    return "high"

def get_cibil_bucket(score: int) -> str:
    if score == -1:    return "no_history"
    if score >= 750:   return "excellent"
    if score >= 700:   return "good"
    if score >= 650:   return "risky"
    return "high_risk"

def engineer_features(inp: dict, feature_cols: list) -> pd.DataFrame:
    """Build the full feature vector for a new customer from basic inputs."""
    age     = inp["age"]
    occ     = inp["occupation"]
    inc     = inp["monthly_income"]
    cibil   = inp["cibil_score"]
    foir    = inp["foir"]
    dti     = inp["dti_ratio"]
    sav     = inp["savings_rate"]
    dig     = inp["digital_adoption_score"]
    inv     = inp["investment_intent_score"]
    onl     = inp["online_spend_ratio"]
    trv     = inp["travel_spend_ratio"]
    rep     = inp["repayment_status"]
    bur     = inp["bureau_inquiries_6m"]
    wof     = inp["writeoff_flag"]
    dfl     = inp["default_history_flag"]
    cit     = inp["city_tier"]
    gender  = inp.get("gender", "M")
    has_hl  = inp.get("home_loan_flag",      0)
    has_cl  = inp.get("car_loan_flag",       0)
    has_pl  = inp.get("personal_loan_flag",  0)
    has_bl  = inp.get("business_loan_flag",  0)
    has_gl  = inp.get("gold_loan_flag",      0)
    has_el  = inp.get("education_loan_flag", 0)
    has_cc  = inp.get("credit_card",         0)
    has_ins = inp.get("insurance",           0)
    has_cd  = inp.get("consumer_durable",    0)

    tenure   = inp.get("tenure_years", 0)
    bal_mult = inp.get("balance_mult", 2.0)
    cc_util  = inp.get("cc_util_pct", 50) if has_cc else 10
    fuel     = inp.get("fuel_spend_ratio", 0.08)
    edu      = inp.get("education_spend_ratio", 0.10)

    ig     = get_income_group(inc)
    cibil_safe = max(cibil, 300)
    cb     = get_cibil_bucket(cibil)
    loan_count = has_hl + has_cl + has_pl + has_bl + has_gl + has_el
    emi    = inc * foir
    bal    = inc * bal_mult
    spend  = inc * (1 - sav)
    upi    = dig * 15

    # CIBIL sub-scores by bucket
    _cs = {"excellent":(92,88,80,75,90,87),"good":(78,72,65,60,75,72),
           "risky":(55,50,48,45,55,52),"high_risk":(30,30,25,25,30,28),"no_history":(0,0,0,0,0,0)}
    ph,us,hs,ms,iq,comp = _cs.get(cb, (50,50,50,50,50,50))

    repay_ord = {"default":0,"delayed":1,"regular":2}[rep]
    dpd = "0" if rep=="regular" else ("1-30" if rep=="delayed" else "61-90")
    dpd_ord = {"0":0,"1-30":1,"31-60":2,"61-90":3,"90+":4}[dpd]
    dti_b   = "severe" if dti>=5 else ("high" if dti>=3 else ("moderate" if dti>=1.5 else "low"))
    ig_ord  = {"low":0,"lower_mid":1,"mid":2,"upper_mid":3,"high":4}[ig]
    cb_ord  = {"no_history":0,"high_risk":1,"risky":2,"good":3,"excellent":4}[cb]
    cb_tier = {"no_history":-1,"high_risk":0,"risky":1,"good":2,"excellent":3}[cb]
    dti_ord = {"low":0,"moderate":1,"high":2,"severe":3}[dti_b]
    cit_enc = max(1, 4 - cit)
    occ_map = {"business":0,"other":1,"retired":2,"salaried":3,"self_employed":4,"student":5}
    gen_map = {"F":0,"M":1,"O":2}
    edu_map = {"below_10th":0,"graduate":1,"high_school":2,"post_graduate":3,"professional":4}
    emp_map = {"government":0,"ngo":1,"other":2,"private":3,"self_employed":4}
    seg = ("youth" if age<30 else
           ("elite" if (ig=="high" and age<45) else ("hni" if ig=="high" else
           ("mass_affluent" if ig=="upper_mid" else ("retail" if ig=="mid" else "mass")))))
    seg_map = {"elite":0,"hni":1,"mass":2,"mass_affluent":3,"retail":4,"youth":5}
    if has_cc:
        cc_var = (4 if (ig in ("upper_mid","high") and cb=="excellent") else
                  3 if trv>=0.20 else 2 if (onl>=0.25 or dig>=2) else 1)
    else:
        cc_var = 0
    repay_pts  = {"regular":25,"delayed":10,"default":0}[rep]
    cibil_pts  = min(max(cibil_safe-550,0)/350.0,1.0)*20
    prod_pts   = min((loan_count + has_cc + has_ins + has_cd)/5.0,1.0)*25
    tenure_pts = min(tenure/15.0,1.0)*30          # was missing — 30 pts max at 15 yrs tenure
    clv_score  = round(tenure_pts + repay_pts + cibil_pts + prod_pts, 2)  # max 100
    clv_enc    = 2 if clv_score>=80 else (1 if clv_score>=72 else 0)      # 98th/88th pct thresholds

    # CC financials — training: cc_monthly_spend = monthly_spend × card_pct[dig]
    # card_pct midpoints: dig0→0.15, dig1→0.275, dig2→0.45, dig3→0.625
    cc_limit = inc * 5.0 if has_cc else 0
    _card_pct = {0: 0.15, 1: 0.275, 2: 0.45, 3: 0.625}.get(min(dig, 3), 0.275)
    cc_spend = spend * _card_pct if has_cc else 0   # fraction of monthly_spend, not util%×limit
    cc_ratio = cc_spend / (cc_limit + 1) if has_cc else 0

    # risk_category_ord: OrdinalEncoder(['high','medium','low']) → high=0, medium=1, low=2
    _risk_ord = (0 if (wof==1 or dfl==1 or cibil_safe<650 or rep=="default")
                 else 2 if (cibil_safe>=750 and rep=="regular" and bur<3)
                 else 1)

    # dominant_spend_enc: LabelEncoder alphabetical → ecommerce=0, education=1, fuel=2,
    # insurance=3, travel=4, utility=5  (67% ecommerce, 22% education, 4% travel in training)
    _dom_enc = (0 if onl>=0.12 else 4 if trv>=0.15 else 1 if inv>=1 else 0)

    f = {
        # --- raw ---
        "age":age, "monthly_income":inc, "cibil_score":cibil_safe,
        "foir":foir, "dti_ratio":dti, "savings_rate":sav,
        "digital_adoption_score":dig, "investment_intent_score":inv,
        "online_spend_ratio":onl, "travel_spend_ratio":trv,
        "spend_to_income_ratio":round(1 - sav, 4), "bureau_inquiries_6m":bur,
        "writeoff_flag":wof, "default_history_flag":dfl,
        "city_tier":cit, "tenure_years":tenure, "salary_account_flag":1 if occ=="salaried" else 0,
        "avg_monthly_balance":bal, "avg_monthly_spend":spend,
        "ecommerce_spend":spend*onl, "education_spend":spend*edu,
        "fuel_spend":spend*fuel, "travel_spend":spend*trv,
        "insurance_premium_spend":spend*0.05, "utility_bill_payments":spend*0.07,
        "emi_amount":emi, "loan_amount":emi/0.022 if emi>0 else 0,
        "outstanding_balance":emi*24 if emi>0 else 0, "loan_tenure_months":36 if emi>0 else 0,
        "upi_txn_count":upi, "debit_card_usage":10 if dig>=1 else 2,
        "monthly_credit_amount":inc*1.05, "monthly_debit_amount":inc*0.75,
        "internet_banking_flag":1 if dig>=2 else 0,
        "mobile_banking_flag":1 if dig>=1 else 0,
        "savings_account_flag":1, "current_account_flag":1 if occ in ("business","self_employed") else 0,
        "fd_flag":1 if sav>=0.30 else 0, "rd_flag":1 if inv>=1 else 0, "debit_card":1,
        "demat_account":1 if inv>=2 else 0,
        "investment_txn_flag":1 if inv>=1 else 0,
        "mutual_fund_txn_flag":1 if inv>=2 else 0,
        "rent_payment_flag":1 if age>=22 else 0,
        "credit_utilization_pct":cc_util,
        "credit_history_years":max(0, age - 22 + tenure // 2) if occ != "student" else 0,
        "missed_payments_count":3 if rep=="delayed" else (6 if rep=="default" else 0),
        "settled_flag":0, "loan_lifecycle_ratio":0,
        "active_loan_count":loan_count, "closed_loan_count":max(0, tenure//4),
        "total_credit_limit":cc_limit,
        "cash_deposit_frequency":2, "foreclosure_flag":0, "topup_loan_flag":0,
        "home_loan_flag":has_hl, "car_loan_flag":has_cl, "education_loan_flag":has_el,
        "personal_loan_flag":has_pl, "business_loan_flag":has_bl, "gold_loan_flag":has_gl,
        "credit_card":has_cc, "insurance":has_ins, "consumer_durable":has_cd,
        "has_any_loan":1 if loan_count>0 else 0,
        "cc_credit_limit":cc_limit,
        "cc_monthly_spend":cc_spend,
        "payment_history_score":ph, "utilization_score":us,
        "credit_history_score":hs, "credit_mix_score":ms,
        "inquiry_score":iq, "cibil_composite":comp,
        "state_avg_income":inc, "clv_score":clv_score,
        "top2_confidence_score":50.0, "top2_risk_percentage":15.0,
        "dominant_spend_enc":_dom_enc, "risk_category_ord":_risk_ord,
        # --- ratios ---
        "balance_to_income_ratio":bal/(inc+1),
        "ecommerce_to_spend_ratio":spend*onl/(spend+1),
        "emi_to_income_ratio":emi/(inc+1),
        "debit_upi_ratio":(10 if dig>=1 else 2)/(upi+1),
        "education_to_income_ratio":spend*edu/(inc+1),
        "balance_months_coverage":bal/(emi+1),
        "credit_utilization_norm":cc_util/100.0,
        # --- composites ---
        "loan_count":loan_count,
        "digital_score":(1 if dig>=2 else 0)+(1 if dig>=1 else 0)+upi/20,
        "investment_appetite":(1 if inv>=1 else 0)+(1 if inv>=2 else 0)+(1 if inv>=2 else 0),
        "liability_breadth":1+(1 if occ in ("business","self_employed") else 0)+(1 if sav>=0.30 else 0),
        "cibil_income_score":cibil_safe*inc/1_000_000,
        "cibil_weighted_composite":ph*0.35+us*0.30+hs*0.15+ms*0.10+iq*0.10,
        # --- interactions ---
        "age_x_cibil":age*cibil_safe/100_000,
        "income_x_any_loan":inc*(1 if loan_count>0 else 0),
        "foir_x_cibil":foir*cibil_safe/1_000,
        "upi_x_income":upi*inc/1_000_000,
        "bureau_x_writeoff":bur*(wof+1),
        "tenure_x_loans":tenure*loan_count, "spend_x_cibil":(1-sav)*cibil_safe/1_000,
        "age_x_income":age*inc/1_000_000,
        "dti_x_foir":dti*foir, "dti_x_cibil":dti*cibil_safe/1_000,
        "savings_x_cibil":sav*cibil_safe/1_000,
        "savings_x_income":sav*inc/100_000,
        "digital_x_online":dig*onl, "digital_x_travel":dig*trv,
        "invest_intent_x_income":inv*inc/100_000,
        "cc_utilization_ratio":cc_ratio,
        "city_digital_score":(4-cit)*dig, "lifecycle_x_cibil":0,
        # --- boundary ---
        "cibil_no_history":1 if cibil==-1 else 0,
        "is_student":1 if occ=="student" else 0,
        "is_retired":1 if occ=="retired" else 0,
        "is_business_occ":1 if occ in ("self_employed","business") else 0,
        "cibil_gte_750":1 if cibil_safe>=750 else 0,
        "cibil_gte_700":1 if cibil_safe>=700 else 0,
        "cibil_gte_650":1 if cibil_safe>=650 else 0,
        "cibil_risky_zone":1 if 650<=cibil_safe<700 else 0,
        "cibil_high_risk":1 if cibil_safe<650 else 0,
        "foir_safe":1 if foir<0.35 else 0, "foir_high":1 if foir>=0.45 else 0,
        "is_blocked":1 if (wof==1 or dfl==1) else 0,
        "prime_home_age":1 if 25<=age<=55 else 0,
        "affluent":1 if ig in ("upper_mid","high") else 0,
        "low_income":1 if ig in ("low","lower_mid") else 0,
        "bureau_blocked":1 if bur>=6 else 0,
        "dti_severe":1 if dti>=5 else 0, "dti_high":1 if 3<=dti<5 else 0,
        "high_saver":1 if sav>=0.30 else 0,
        "strong_digital":1 if dig>=2 else 0,
        "travel_spender":1 if trv>=0.15 else 0,
        "online_spender":1 if onl>=0.12 else 0,
        "tier1_city":1 if cit==1 else 0, "tier1_or_2_city":1 if cit<=2 else 0,
        "invest_intent_high":1 if inv>=2 else 0,
        "profit_of_top_eligible":75,
        "credit_capacity_score":cibil_safe/10-foir*50-dti*5+{"low":0,"lower_mid":5,"mid":10,"upper_mid":15,"high":20}[ig],
        "upi_power_user":1 if upi>=30 else 0,
        "upi_frequency_bucket":0 if upi==0 else (1 if upi<=9 else (2 if upi<=29 else 3)),
        "ecommerce_intensity":min(spend*onl/(inc+1),1.0),
        "digital_credit_ready":1 if (dig>=2 and cibil_safe>=700) else 0,
        "non_cash_ratio":min((1 if upi>0 else 0)+has_cc+1, 3),
        "young_edu_candidate":1 if (17<=age<=35 and occ!="student" and not has_el) else 0,
        "age_edu_window":max(0,1.0-abs(age-26)/10.0) if 17<=age<=35 else 0.0,
        # --- ordinal ---
        "income_group_ord":ig_ord, "cibil_score_bucket_ord":cb_ord,
        "dpd_bucket_ord":dpd_ord, "repayment_status_ord":repay_ord,
        "dti_bucket_ord":dti_ord, "cibil_tier_enc":cb_tier,
        "city_tier_enc":cit_enc, "cc_variant_enc":cc_var,
        # --- label encoded ---
        "occupation_enc":occ_map.get(occ,3),
        "gender_enc":gen_map.get(gender,1),
        "marital_status_enc":{"divorced":0,"married":1,"single":2,"widowed":3}.get(inp.get("marital_status","single"),2),
        "employer_type_enc":emp_map.get(inp.get("employer_type","private"),3),
        "education_level_enc":edu_map.get(inp.get("education_level","graduate"),1),
        "customer_segment_enc":seg_map.get(seg,2),
        "clv_tier_enc":clv_enc, "is_premium":1 if clv_score>=72 else 0,
    }

    df = pd.DataFrame([f])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols].fillna(0)

def predict_for_new_customer(inp: dict):
    """Run full ML pipeline for a new customer and return recommendation dict."""
    if clf_model is None:
        return None, "Models not loaded"

    ig   = get_income_group(inp["monthly_income"])
    age  = inp["age"]
    occ  = inp["occupation"]
    foir = inp["foir"]
    inc  = inp["monthly_income"]
    cibil = inp["cibil_score"]
    cibil_safe = max(cibil, 300)

    feat_df = engineer_features(inp, feature_cols_model)
    proba_xgb = clf_model.predict_proba(feat_df)
    proba_raw = (0.55 * proba_xgb + 0.45 * clf_cb_model.predict_proba(feat_df)
                 if clf_cb_model is not None else proba_xgb)
    classes = le_model.classes_

    # Build valid scores with gates + utility
    proba_by_product = {prod: float(proba_raw[0][i]) for i, prod in enumerate(classes)}

    valid_scores = {}
    for i, prod in enumerate(classes):
        if prod not in PRODUCT_PROFIT:
            continue
        if not passes_all_gates(prod, age, occ, ig, inc, foir):
            continue
        conf_norm   = float(proba_raw[0][i])
        profit_norm = PRODUCT_PROFIT[prod] / 100.0
        risk_score  = compute_bank_risk_score_simple(prod, cibil_safe, get_cibil_bucket(cibil),
                                                      inp["repayment_status"], inp["writeoff_flag"],
                                                      inp["default_history_flag"],
                                                      inp["bureau_inquiries_6m"], foir,
                                                      inp["dti_ratio"], age, occ,
                                                      inp["savings_rate"])
        utility = (conf_norm * 0.45 + profit_norm * 0.35 - (risk_score/100) * 0.20) \
                  * PRODUCT_SATURATION.get(prod, 1.0)
        valid_scores[prod] = max(0, utility)

    if not valid_scores:
        return None, "No eligible products found for this customer profile."

    ranked = sorted(valid_scores.items(), key=lambda x: -x[1])
    r1_prod, r1_util = ranked[0]
    r1_conf = round(float(proba_raw[0][list(classes).index(r1_prod)]) * 100, 1) if r1_prod in classes else 0.0
    r2_prod = ranked[1][0] if len(ranked) > 1 else "insurance"
    r2_conf = round(float(proba_raw[0][list(classes).index(r2_prod)]) * 100, 1) if r2_prod in classes else 0.0

    # Loan ranges
    r1_ranges = compute_loan_range(r1_prod, ig, age, inc, foir)
    r2_ranges = compute_loan_range(r2_prod, ig, age, inc, foir)

    r1_risk = compute_bank_risk_score_simple(r1_prod, cibil_safe, get_cibil_bucket(cibil),
                                              inp["repayment_status"], inp["writeoff_flag"],
                                              inp["default_history_flag"], inp["bureau_inquiries_6m"],
                                              foir, inp["dti_ratio"], age, occ, inp["savings_rate"])
    r2_risk = compute_bank_risk_score_simple(r2_prod, cibil_safe, get_cibil_bucket(cibil),
                                              inp["repayment_status"], inp["writeoff_flag"],
                                              inp["default_history_flag"], inp["bureau_inquiries_6m"],
                                              foir, inp["dti_ratio"], age, occ, inp["savings_rate"])

    # Confidence from utility ratio when ML conf is low
    if r1_conf < 1.0:
        total = sum(v for v in valid_scores.values())
        r1_conf = round(min((r1_util / max(total, 0.001)) * 85, 85), 1)
        r2_conf = round(min((valid_scores.get(r2_prod,0) / max(total,0.001)) * 85, 85), 1)

    return {
        "r1_prod": r1_prod, "r1_conf": r1_conf, "r1_ranges": r1_ranges, "r1_risk": r1_risk,
        "r2_prod": r2_prod, "r2_conf": r2_conf, "r2_ranges": r2_ranges, "r2_risk": r2_risk,
        "valid_scores": valid_scores, "income_group": ig,
        "proba_by_product": proba_by_product,
    }, None

def compute_bank_risk_score_simple(product, _cibil_score, cibil_bucket, repayment,
                                    writeoff, default_hist, bureau, foir, dti, _age, _occ, savings_rate):
    if writeoff == 1:        return 95.0
    if default_hist == 1:    return 90.0
    base = {"home_loan":20,"car_loan":22,"personal_loan":38,"education_loan":25,
            "gold_loan":12,"business_loan":42,"credit_card":28,"insurance":5,"consumer_durable":30}.get(product,25)
    cadj = {"excellent":-18,"good":-8,"risky":+15,"high_risk":+30,"no_history":+12}.get(cibil_bucket,0)
    radj = {"regular":-8,"delayed":+10,"default":+25}.get(repayment,0)
    fadj = (+20 if foir>=0.50 else +12 if foir>=0.40 else +5 if foir>=0.30 else +2 if foir>=0.20 else -3)
    dadj = (+18 if dti>=5 else +10 if dti>=3 else +4 if dti>=1.5 else -2)
    badj = min(bureau*4, 20)
    income_adj = 0  # income adjustment handled via cibil_bucket already
    sadj = -5 if (product=="insurance" and savings_rate>=0.30) else 0
    return round(float(np.clip(base+cadj+radj+fadj+dadj+badj+sadj, 0, 100)), 1)

def passes_all_gates(prod: str, age: int, occ: str, ig: str,
                     monthly_income: float, foir: float) -> bool:
    """Returns True only if product passes age, occupation, and affordability gates."""
    if occ == "student" and prod in STUDENT_BLOCKED:
        return False
    gates = AGE_GATES.get(prod, {})
    if "max_age" in gates and age > gates["max_age"]:
        return False
    if "min_age" in gates and age < gates["min_age"]:
        return False
    if not is_affordable(prod, ig, monthly_income, foir):
        return False
    # Business loan: income + FOIR gate (low/lower_mid income → personal_loan instead)
    if prod == "business_loan":
        if ig in ("low", "lower_mid"):
            return False
        if ig == "mid" and foir >= 0.35:
            return False
        if foir >= 0.35:
            return False
    return True

def get_validated_recommendations(rec: pd.Series, cust: pd.Series):
    """
    Returns rank-1 and rank-2 that both pass ALL gates (age, occupation, affordability).
    If the CSV rank-1 fails any gate, it is overridden with the best valid product.

    Returns:
        r1, r1_conf, r1_ranges, r2, r2_conf, r2_ranges, original_r1, was_overridden
    """
    age    = int(cust["age"])
    occ    = str(cust["occupation"])
    ig     = str(rec.get("income_group", "mid"))
    inc    = float(rec.get("monthly_income", 50_000))
    foir   = float(rec.get("foir", 0.0))
    cibil  = int(rec.get("cibil_score", 300))
    csv_r1 = rec["ml_rank1_recommendation"]

    # ── No CIBIL history (NH) — new first-time credit user ──────
    # In real India banking, CIBIL = -1 means no credit history.
    # Education loan + insurance are always available regardless of CIBIL.
    if cibil <= 0:
        ig_safe   = ig if ig in LOAN_AMOUNT_RANGES.get("education_loan", {}) else "mid"
        r1_ranges = compute_loan_range("education_loan", ig_safe, age, inc, foir)
        r2_ranges = compute_loan_range("insurance", ig_safe, age, inc, foir)
        return ("education_loan", 75.0, r1_ranges,
                "insurance",      50.0, r2_ranges,
                csv_r1, cibil != int(rec.get("cibil_score", 300)))

    score_cols = [c for c in rec.index if c.startswith("score_")]

    # Score all valid products
    valid_scores = {}
    for col in score_cols:
        prod  = col.replace("score_", "")
        score = float(rec[col])
        if score <= 0:
            continue
        if not passes_all_gates(prod, age, occ, ig, inc, foir):
            continue
        mult = AGE_PRODUCT_MULTIPLIER.get(prod, lambda _: 1.0)(age)
        valid_scores[prod] = score * mult

    # Fallback: insurance or gold_loan if nothing valid
    if not valid_scores:
        fallback = "insurance"
        ranges   = compute_loan_range(fallback, ig, age, inc, foir)
        return fallback, 0.0, ranges, "gold_loan", 0.0, compute_loan_range("gold_loan", ig, age, inc, foir), csv_r1, True

    # Sort by score descending
    ranked = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

    # Rank-1
    r1_prod, r1_score = ranked[0]
    was_overridden    = (r1_prod != csv_r1)
    csv_r1_conf       = float(rec.get("rank1_confidence_pct", 0.0))
    total_util        = max(sum(valid_scores.values()), 0.001)

    # Confidence: prefer CSV confidence when valid; fall back to utility-score ratio
    if not was_overridden and csv_r1_conf >= 1.0:
        r1_conf = csv_r1_conf
    else:
        # Utility-based confidence — how dominant is rank-1 among valid products
        util_conf = round(min((r1_score / total_util) * 85.0, 85.0), 1)
        r1_conf   = max(util_conf, 5.0)  # minimum 5% so it's never 0.0%

    r1_ranges = compute_loan_range(r1_prod, ig, age, inc, foir)

    # Rank-2: best valid product that is neither rank-1 NOR the overridden product
    r2_candidates = [(p, s) for p, s in ranked if p != r1_prod and p != csv_r1]

    if not r2_candidates:
        # Try safe fallbacks that aren't rank-1 or overridden product
        for candidate in ["insurance", "gold_loan", "consumer_durable", "personal_loan"]:
            if (candidate != r1_prod and candidate != csv_r1
                    and passes_all_gates(candidate, age, occ, ig, inc, foir)):
                fb_ranges = compute_loan_range(candidate, ig, age, inc, foir)
                return r1_prod, r1_conf, r1_ranges, candidate, 5.0, fb_ranges, csv_r1, was_overridden
        # Absolute fallback: just use rank-1 info with a note
        return r1_prod, r1_conf, r1_ranges, r1_prod, 0.0, r1_ranges, csv_r1, was_overridden

    r2_prod, r2_score = r2_candidates[0]
    r2_conf   = round(min((r2_score / max(r1_score, 0.001)) * r1_conf, 95.0), 1)
    r2_conf   = max(r2_conf, 5.0)
    r2_ranges = compute_loan_range(r2_prod, ig, age, inc, foir)

    return r1_prod, r1_conf, r1_ranges, r2_prod, r2_conf, r2_ranges, csv_r1, was_overridden

FEATURE_DESCRIPTIONS = {
    "insurance"              : "customer does NOT currently own insurance (protection gap)",
    "credit_card"            : "customer does NOT currently own a credit card",
    "consumer_durable"       : "consumer durable loan ownership flag",
    "home_loan_flag"         : "existing home loan ownership",
    "gold_loan_flag"         : "existing gold loan ownership",
    "personal_loan_flag"     : "existing personal loan ownership",
    "monthly_income"         : "customer's monthly income in ₹",
    "affluent"               : "income group is upper_mid or high (1=yes, 0=no)",
    "low_income"             : "income group is low or lower_mid (1=yes, 0=no)",
    "cibil_score"            : "CIBIL credit score 300–900 (higher = better creditworthiness)",
    "cibil_gte_750"          : "CIBIL ≥750 (excellent tier, 1=yes) — eligible for all products",
    "cibil_gte_700"          : "CIBIL ≥700 (good tier, 1=yes) — eligible for most products",
    "cibil_tier_enc"         : "CIBIL tier: 0=high_risk, 1=risky, 2=good, 3=excellent",
    "cibil_income_score"     : "CIBIL × income interaction (higher = creditworthy AND wealthy)",
    "foir"                   : "FOIR — EMI-to-income ratio (lower = more repayment capacity)",
    "foir_safe"              : "FOIR <35% (1=yes) — significant EMI capacity available",
    "foir_high"              : "FOIR ≥45% (1=yes) — customer already heavily leveraged",
    "dti_ratio"              : "Debt-to-Income ratio (total debt vs annual income)",
    "dti_severe"             : "DTI ≥5x (1=yes) — severely over-leveraged",
    "savings_rate"           : "monthly savings as % of income (higher = financially disciplined)",
    "high_saver"             : "savings rate ≥30% (1=yes) — strong saver, suitable for investments",
    "savings_x_cibil"        : "high savings × good CIBIL interaction",
    "investment_intent_score": "investment activity score 0–3 (MF, demat, investment transactions)",
    "invest_intent_high"     : "investment intent ≥2 (1=yes) — actively investing customer",
    "spend_to_income_ratio"  : "monthly spend ÷ income (higher = more spender)",
    "online_spend_ratio"     : "online/ecommerce spend as % of total spend",
    "travel_spend_ratio"     : "travel spend as % of total spend",
    "travel_spender"         : "travel spend ≥15% (1=yes) — frequent traveller",
    "online_spender"         : "online spend ≥12% (1=yes) — active online shopper",
    "digital_adoption_score" : "digital banking score 0–3 (UPI + mobile + internet banking)",
    "strong_digital"         : "digital adoption ≥2 (1=yes) — active digital banking user",
    "upi_txn_count"          : "monthly UPI transaction count",
    "upi_power_user"         : "UPI ≥30/month (1=yes) — heavy digital payments user",
    "age"                    : "customer age in years",
    "prime_home_age"         : "age 25–55 (1=yes) — prime window for a home loan",
    "is_student"             : "occupation is student (1=yes)",
    "is_retired"             : "occupation is retired (1=yes)",
    "is_business_occ"        : "occupation is self-employed or business owner (1=yes)",
    "clv_score"              : "Customer Lifetime Value score 0–100",
    "clv_tier_enc"           : "CLV tier: 0=normal, 1=elite, 2=super_hni",
    "is_premium"             : "CLV tier is elite or super_hni (1=yes)",
    "balance_to_income_ratio": "average balance ÷ monthly income (higher = more financially stable)",
    "credit_capacity_score"  : "composite credit headroom (CIBIL, FOIR, DTI, income)",
    "tenure_years"           : "years as bank customer",
}

def explain_feature(feat: str, value: float, shap_val: float) -> str:
    desc   = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_"," "))
    val_str = f"{value:.0f}" if value == int(value) else f"{value:.3f}"
    if shap_val > 0.3:   direction = "**strongly supports** this recommendation"
    elif shap_val > 0:   direction = "supports this recommendation"
    else:                direction = "works against this recommendation"
    return f"{desc} = **{val_str}** → {direction}"

def build_context(customer_id: str, validated_r1: str, validated_r1_conf: float,
                  r1_ranges: tuple, validated_r2: str,
                  was_overridden: bool, original_r1: str) -> str:
    rec  = recs_df.loc[customer_id]
    cust = customers_df.loc[customer_id]

    a1,a2,t1,t2,e1,e2 = r1_ranges
    amt_str    = f"{fmt(a1)} – {fmt(a2)}" if a2 > 0 else "N/A"
    tenure_str = f"{t1}–{t2} months"      if t2 > 0 else "N/A"
    emi_str    = f"{fmt(e1)} – {fmt(e2)}/month" if e2 > 0 else "N/A"
    conf_qual  = "high" if validated_r1_conf >= 70 else ("medium" if validated_r1_conf >= 40 else "low")
    risk_label = str(rec.get(f"bank_risk_label_{validated_r1}", "Low"))
    risk_score = float(rec.get(f"bank_risk_score_{validated_r1}", 0))

    # Compute explicit FOIR status so LLM cannot misread threshold notation
    foir_val = float(rec['foir'])
    if foir_val < 0.35:
        foir_status = f"SAFE ({foir_val:.1%} — below 35% safe threshold, good repayment capacity)"
    elif foir_val < 0.45:
        foir_status = f"MODERATE ({foir_val:.1%} — above safe 35%, approaching high 45%, limited capacity)"
    else:
        foir_status = f"HIGH ({foir_val:.1%} — above 45%, already heavily leveraged, risky to add more EMI)"

    override_note = ""
    if was_overridden:
        headroom = max(0, 0.50 - foir_val)
        override_note = f"""
⚠️ RECOMMENDATION WAS OVERRIDDEN — YOU MUST MENTION THIS:
- Original ML recommendation: {original_r1.replace('_',' ').upper()}
- Reason rejected: FOIR headroom is only {headroom:.1%}, which cannot cover the minimum EMI for {original_r1.replace('_',' ')}
- Replaced with: {validated_r1.replace('_',' ').upper()} (next best valid and affordable product)
- When explaining this recommendation, START by stating that {original_r1.replace('_',' ')} was the ML's original choice but was overridden due to affordability constraints.
"""

    # No-CIBIL note for LLM context
    no_cibil_note = ""
    if int(rec.get("cibil_score", 300)) <= 0:
        no_cibil_note = """
⚠️ NO CIBIL HISTORY (NH): This customer is a new/first-time credit user with no CIBIL score.
In India, education loans do NOT require CIBIL — they are based on academic merit and a guarantor.
Insurance also requires no CIBIL check.
Do NOT mention CIBIL score as a positive factor for this customer. Explain the recommendation based on their student status and future earning potential."""

    shap_text = ""
    if customer_id in shap_df.index:
        s = shap_df.loc[customer_id]
        shap_text = f"""
TOP 3 REASONS FROM ML MODEL (SHAP — explain in plain banking language, never repeat raw names):
  1. [{s['top1_feature']}] {explain_feature(s['top1_feature'], s['top1_value'], s['top1_shap'])}
  2. [{s['top2_feature']}] {explain_feature(s['top2_feature'], s['top2_value'], s['top2_shap'])}
  3. [{s['top3_feature']}] {explain_feature(s['top3_feature'], s['top3_value'], s['top3_shap'])}
MAIN BLOCKER: [{s['top_blocker_feature']}] {explain_feature(s['top_blocker_feature'], s['top_blocker_shap'], s['top_blocker_shap'])}"""
    else:
        shap_text = """
SHAP DATA: NOT AVAILABLE for this customer.
⛔ DO NOT fabricate or guess SHAP feature values.
⛔ DO NOT present made-up "SHAP insights" or feature breakdowns.
When asked why this product was recommended, explain ONLY using the financial profile data above."""

    return f"""
=== CUSTOMER PROFILE ===
ID: {customer_id} | Age: {int(cust['age'])} | Gender: {cust['gender']} | Marital: {cust['marital_status']}
Occupation: {cust['occupation']} ({cust['employer_type']}) | Education: {cust['education_level']}
Location: {cust['city']}, {cust['state']} (Tier-{int(rec['city_tier'])} city)
Bank Tenure: {int(cust['tenure_years'])} years | Segment: {cust['customer_segment']} | CLV: {cust['clv_tier']}

=== FINANCIAL PROFILE ===
Monthly Income : ₹{rec['monthly_income']:,.0f} ({rec['income_group']})
CIBIL Score    : {int(rec['cibil_score'])} — {rec['cibil_score_bucket']}
Repayment      : {rec['repayment_status']}
FOIR           : {foir_status}
DTI Ratio      : {rec['dti_ratio']:.2f}x ({rec['dti_bucket']})
Savings Rate   : {rec['savings_rate']:.1%}
Digital Score  : {int(rec['digital_adoption_score'])}/3
Online Spend   : {rec['online_spend_ratio']:.1%} | Travel Spend: {rec['travel_spend_ratio']:.1%}
{override_note}{no_cibil_note}
=== VALIDATED RECOMMENDATION ===
Rank-1: {validated_r1.replace('_',' ').upper()} | Confidence: {validated_r1_conf:.1f}% ({conf_qual}) | Bank Risk: {risk_score:.0f}/100 ({risk_label})
Amount: {amt_str} | Tenure: {tenure_str} | EMI: {emi_str}
Rank-2 (alternative): {validated_r2.replace('_',' ').upper()}
{shap_text}
"""

# ── Session state ─────────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages = []
if "current_customer" not in st.session_state: st.session_state.current_customer = None
if "nc_inp"           not in st.session_state: st.session_state.nc_inp = None
if "nc_result"        not in st.session_state: st.session_state.nc_result = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ICICI Bank")
    st.markdown("### Product Recommendation System")
    st.caption("ML Pipeline v5.1 · Data Generator v6.1")
    st.divider()

    raw_id      = st.text_input("Enter Customer ID", placeholder="e.g. CUST000001")
    customer_id = raw_id.upper().strip()

    if customer_id and customer_id != st.session_state.current_customer:
        st.session_state.current_customer = customer_id
        st.session_state.messages = []

    if customer_id and customer_id in recs_df.index:
        rec  = recs_df.loc[customer_id]
        cust = customers_df.loc[customer_id]
        st.success("✅ Customer found")
        st.metric("CIBIL Score",    int(rec["cibil_score"]),                rec["cibil_score_bucket"])
        st.metric("Monthly Income", f"₹{rec['monthly_income']/1000:.0f}K", rec["income_group"])
        st.metric("CLV Tier",       cust["clv_tier"].upper())
        st.metric("Repayment",      rec["repayment_status"].capitalize())
    elif customer_id:
        st.error(f"❌ {customer_id} not found. Valid range: CUST000001–CUST020000")

    st.divider()
    st.markdown("**Models**")
    st.caption("🤖 XGBoost + CatBoost Ensemble")
    st.caption("🌳 Decision Tree (explainability)")
    st.caption("🔍 SHAP (feature attribution)")
    st.caption("🧠 Llama 3.1 8B via Groq")

# ── Always show 3 tabs ────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 AI Chatbot", "🆕 New Customer", "🔍 Pipeline"])

# Shared variables for tab1 and tab2
rec = cust = r1 = r2 = r1_conf = r2_conf = None
r1_a1=r1_a2=r1_t1=r1_t2=r1_e1=r1_e2=0
r2_a1=r2_a2=r2_t1=r2_t2=r2_e1=r2_e2=0
original_r1 = was_overridden = None
warnings = []

if customer_id and customer_id in recs_df.index:
    rec  = recs_df.loc[customer_id]
    cust = customers_df.loc[customer_id]
    (r1, r1_conf, (r1_a1,r1_a2,r1_t1,r1_t2,r1_e1,r1_e2),
     r2, r2_conf, (r2_a1,r2_a2,r2_t1,r2_t2,r2_e1,r2_e2),
     original_r1, was_overridden) = get_validated_recommendations(rec, cust)
    warnings = get_warnings(rec, cust, validated_conf=r1_conf)
else:
    with tab1:
        st.title("🏦 ICICI Bank — Product Recommendation System")
        if customer_id:
            st.error(f"Customer **{customer_id}** not found. Try CUST000001 to CUST020000.")
        else:
            st.info("👈 Enter a Customer ID in the sidebar to view their dashboard.")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(recs_df):,}")
        c2.metric("Products", "9")
        c3.metric("ML Models", "3")
        c4.metric("Features", "100+")
    with tab2:
        st.info("👈 Enter a Customer ID in the sidebar to use the chatbot.")

if customer_id and customer_id in recs_df.index:
    # ── TAB 1: DASHBOARD ──────────────────────────────────────
    with tab1:
        st.title(f"Customer Dashboard — {customer_id}")

        # Override banner — shown before anything else if rank-1 was replaced
        if was_overridden:
            st.error(
                f"🔄 **Recommendation Overridden:** The ML model originally recommended "
                f"**{original_r1.replace('_',' ').title()}**, but it failed policy gates "
                f"(age, occupation, or affordability). Showing next best valid product instead."
            )

        # All other warnings
        for level, msg in warnings:
            if level == "error":
                st.error(msg)
            else:
                st.warning(msg)

        # KPIs
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Age",        int(cust["age"]))
        c2.metric("CIBIL",      int(rec["cibil_score"]), rec["cibil_score_bucket"])
        c3.metric("Income/mo",  f"₹{rec['monthly_income']/1000:.0f}K")
        c4.metric("FOIR",       f"{rec['foir']:.1%}")
        c5.metric("Repayment",  rec["repayment_status"].capitalize())

        st.divider()

        # Recommendation cards
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            risk_label = str(rec.get(f"bank_risk_label_{r1}", "Low"))
            risk_score = float(rec.get(f"bank_risk_score_{r1}", 0))
            emoji      = RISK_EMOJI.get(risk_label, "⚪")

            st.subheader("🥇 Primary Recommendation")
            st.markdown(f"## {r1.replace('_',' ').title()}")
            if was_overridden:
                st.caption(f"_(original: {original_r1.replace('_',' ').title()} — overridden)_")

            m1, m2 = st.columns(2)
            conf_delta = "High" if r1_conf >= 70 else ("Medium" if r1_conf >= 40 else "Low ⚠️")
            m1.metric("Confidence", f"{r1_conf:.1f}%", conf_delta)
            m2.metric(f"Bank Risk {emoji}", f"{risk_score:.0f}/100 · {risk_label}")

            if r1_a2 > 0:
                st.markdown(f"💰 **Amount:** {fmt(r1_a1)} – {fmt(r1_a2)}")
                st.markdown(f"📅 **Tenure:** {r1_t1}–{r1_t2} months")
                if r1_e2 > 0: st.markdown(f"📆 **EMI:** {fmt(r1_e1)} – {fmt(r1_e2)}/mo")

        with col_r2:
            r2_risk_label = str(rec.get(f"bank_risk_label_{r2}", "Low"))
            r2_risk_score = float(rec.get(f"bank_risk_score_{r2}", 0))
            r2_emoji      = RISK_EMOJI.get(r2_risk_label, "⚪")

            st.subheader("🥈 Alternative Recommendation")
            st.markdown(f"## {r2.replace('_',' ').title()}")

            m3, m4 = st.columns(2)
            r2_conf_delta = "High" if r2_conf >= 70 else ("Medium" if r2_conf >= 40 else "Low")
            m3.metric("Confidence", f"{r2_conf:.1f}%", r2_conf_delta)
            m4.metric(f"Bank Risk {r2_emoji}", f"{r2_risk_score:.0f}/100 · {r2_risk_label}")

            if r2_a2 > 0:
                st.markdown(f"💰 **Amount:** {fmt(r2_a1)} – {fmt(r2_a2)}")
                st.markdown(f"📅 **Tenure:** {r2_t1}–{r2_t2} months")
                if r2_e2 > 0: st.markdown(f"📆 **EMI:** {fmt(r2_e1)} – {fmt(r2_e2)}/mo")

        st.divider()

        # SHAP explanation
        st.subheader("🔍 Why This Recommendation?")
        if customer_id in shap_df.index:
            s = shap_df.loc[customer_id]
            c1,c2,c3 = st.columns(3)
            for col, rank in [(c1,1),(c2,2),(c3,3)]:
                feat   = s[f"top{rank}_feature"]
                val    = s[f"top{rank}_value"]
                shp    = s[f"top{rank}_shap"]
                desc   = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_"," "))
                val_str = f"{val:.0f}" if val == int(val) else f"{val:.3f}"
                col.info(f"**Reason {rank}**\n\n`{feat}`\n\n{desc}\n\nValue: **{val_str}** · SHAP: **+{shp:.4f}**")

            blocker_desc = FEATURE_DESCRIPTIONS.get(s['top_blocker_feature'], s['top_blocker_feature'].replace('_',' '))
            st.warning(f"⚠️ **Main limiting factor:** `{s['top_blocker_feature']}` — {blocker_desc} (SHAP: {s['top_blocker_shap']:.4f})")
        else:
            st.info("ℹ️ SHAP data available for test-set customers only (20%). For this customer, the recommendation is based on ML utility scores from the financial profile above.")

        st.divider()

        # All products risk table
        st.subheader("🏦 Bank Risk Scores — All 9 Products")
        risk_rows = []
        for p in ALL_PRODUCTS:
            score = float(rec.get(f"bank_risk_score_{p}", 0))
            label = str(rec.get(f"bank_risk_label_{p}", "Low"))
            util  = float(rec.get(f"score_{p}", 0))
            risk_rows.append({
                "Product"      : p.replace("_"," ").title(),
                "Risk Score"   : f"{score:.0f}/100",
                "Risk Label"   : f"{RISK_EMOJI.get(label,'⚪')} {label}",
                "Utility Score": f"{util:.4f}",
            })
        st.dataframe(
            pd.DataFrame(risk_rows).sort_values("Risk Score"),
            use_container_width=True, hide_index=True
        )

    # ── TAB 2: CHATBOT ────────────────────────────────────────
    with tab2:
        st.title(f"💬 Ask About {customer_id}")
        st.caption("Ask anything about this customer's recommendation, risk, eligibility, or loan details.")

        with st.expander("💡 Suggested Questions"):
            st.markdown("""
- Why was this customer recommended this product?
- What is their credit risk and what does it mean?
- Are they eligible for a home loan? Why or why not?
- How much loan can this customer actually afford?
- What is their second best option and why?
- Is this a risky customer for the bank?
- What would improve this customer's recommendation?
            """)

        st.divider()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Ask a question about this customer..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            history = []
            for m in st.session_state.messages[:-1]:
                history.append(HumanMessage(content=m["content"]) if m["role"]=="user"
                               else AIMessage(content=m["content"]))

            system_prompt = f"""You are an expert ICICI Bank relationship manager and AI assistant.

STRICT RULES — FOLLOW ALL OF THESE:
1. FOIR RULE: The FOIR status is explicitly stated in the profile (SAFE / MODERATE / HIGH). Use only that label. NEVER compute or restate FOIR differently. Never say FOIR is "safe" if the profile says MODERATE or HIGH.
2. OVERRIDE RULE: If the context says "RECOMMENDATION WAS OVERRIDDEN", you MUST mention this at the start of your answer. Explain that the original product was rejected and why.
3. CONFIDENCE RULE: If confidence is below 40%, start your answer with a clear statement that this is a low-confidence recommendation.
4. SHAP RULE: If SHAP data is "NOT AVAILABLE", never fabricate SHAP insights or feature values. Explain using only the financial profile facts.
5. FEATURE RULE: For binary flags, value=0 = customer does NOT have it; value=1 = customer DOES have it. Never swap these.
6. FACTUAL RULE: Never state something that contradicts the data. If you are unsure, say so.
7. Use ₹ for amounts. Be concise and specific.

{build_context(customer_id, r1, r1_conf, (r1_a1,r1_a2,r1_t1,r1_t2,r1_e1,r1_e2), r2, was_overridden, original_r1)}"""

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            chain = prompt_template | llm | StrOutputParser()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chain.invoke({"history": history, "question": user_input})
                    except ConnectionResetError:
                        response = (
                            "The chat connection was closed by the remote model provider. "
                            "Please try the same question again."
                        )
                    except Exception as exc:
                        response = f"Unable to get a chatbot response right now: {exc}"
                st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

# ── TAB 3: NEW CUSTOMER (always rendered, no customer ID needed) ──
with tab3:
    st.title("🆕 New Customer Prediction")
    st.caption("Enter a new customer's details to get an instant recommendation. Fields marked ⚡ are most important.")

    if clf_model is None:
        err_detail = _model_load_error or "models not found in models/ folder"
        st.error(f"ML models failed to load: {err_detail}")
        st.warning("Fix: stop the Streamlit server (Ctrl+C) and run `streamlit run chatbot/app.py` again. Refreshing the browser tab is NOT enough — the cache lives in the server process.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("👤 Basic Profile")
            nc_age  = st.slider("Age ⚡", 18, 75, 30,
                                help="Customer's age. Home loan blocked >55, personal loan >65.")
            nc_gender = st.selectbox("Gender", ["M","F","O"])
            nc_occ  = st.selectbox("Occupation ⚡",
                                   ["salaried","self_employed","business","student","retired","other"],
                                   help="Salaried → personal/home loan. Self-employed/business → business loan (if income high). Student → education loan.")
            nc_emp  = st.selectbox("Employer Type",
                                   ["private","government","self_employed","ngo","other"],
                                   help="Government employees get preference for home & personal loans.")
            nc_edu  = st.selectbox("Education",
                                   ["graduate","post_graduate","professional","high_school","below_10th"])
            nc_mar  = st.selectbox("Marital Status", ["single","married","divorced","widowed"])
            nc_cit  = st.radio("City Tier ⚡", [1, 2, 3],
                               horizontal=True,
                               help="1 = Metro (Mumbai/Delhi/Bangalore). 2 = Large city. 3 = Small city/town.")
            nc_tenure = st.slider("Bank Tenure (years) ⚡", 0, 15, 3,
                                  help="How many years has this customer been with the bank? Training data: mean=7 yrs, range 0–15.")

        with col_b:
            st.subheader("💳 Financial Profile")
            nc_inc  = st.number_input("Monthly Income (₹) ⚡", 5_000, 1_000_000, 50_000, step=5_000,
                                      help="Low: ₹5K–15K | Lower-Mid: ₹15K–30K | Mid: ₹30K–60K | Upper-Mid: ₹60K–1.5L | High: ₹1.5L+")
            nc_cibil = st.number_input("CIBIL Score ⚡", -1, 900, 720, step=1,
                                       help="-1 = No History (new credit user). <650 = High Risk. 650-699 = Risky. 700-749 = Good. 750+ = Excellent.")
            nc_rep  = st.selectbox("Repayment History ⚡", ["regular","delayed","default"],
                                   help="Regular = pays on time. Delayed = some late payments. Default = missed payments.")
            nc_foir = st.slider("FOIR (Existing EMI ÷ Income) ⚡", 0.0, 0.80, 0.0, 0.01,
                                help="0.0 = No existing loans. 0.35 = Safe zone. >0.50 = Heavily leveraged, only gold/insurance.")
            nc_dti  = st.slider("DTI Ratio (Total Debt ÷ Annual Income)", 0.0, 10.0, 0.5, 0.1,
                                help="0 = No debt. <1.5 = Low. 1.5-3 = Moderate. 3-5 = High. >5 = Severe (blocks most loans).")
            nc_bur  = st.slider("Bureau Inquiries (last 6 months)", 0, 9, 0,
                                help="Number of loan/card applications in 6 months. ≥6 blocks credit card.")
            nc_wof  = st.checkbox("Has Writeoff / NPA history",
                                  help="If checked, only insurance and FD can be recommended.")
            nc_dfl  = st.checkbox("Has Default History",
                                  help="Prior loan default. Blocks all credit products.")
            nc_bal_mult = st.slider("Avg Monthly Balance (× income) ⚡", 0.5, 8.0, 2.0, 0.1,
                                    help="Average bank balance as a multiple of monthly income. Training data: mean=2.66×, range 0.67×–8.18×. Low balance = financially stretched.")

        st.divider()
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("💰 Financial Behaviour")
            nc_sav  = st.slider("Savings Rate ⚡", 0.0, 0.70, 0.25, 0.01,
                                help="Monthly savings ÷ monthly income. ≥0.30 = good saver → insurance boost.")
            nc_dig  = st.radio("Digital Adoption Score ⚡", [0, 1, 2, 3],
                               horizontal=True,
                               help="0 = No digital banking. 1 = Mobile banking only. 2 = Mobile + UPI. 3 = Full digital (UPI + Mobile + Internet).")
            nc_inv  = st.radio("Investment Intent Score", [0, 1, 2, 3],
                               horizontal=True,
                               help="0 = No investments. 1 = Some. 2 = Active investor. 3 = Heavy investor (MF + demat + regular txns).")
            nc_onl  = st.slider("Online Spend Ratio", 0.0, 0.50, 0.10, 0.01,
                                help="% of spending that's online/ecommerce. ≥0.12 → credit card boost.")
            nc_trv  = st.slider("Travel Spend Ratio", 0.0, 0.40, 0.05, 0.01,
                                help="% of spending on travel. ≥0.15 → travel credit card boost.")
            nc_fuel = st.slider("Fuel Spend Ratio", 0.03, 0.20, 0.08, 0.01,
                                help="% of spending on fuel/transport. RBI data range: 5–12%.")
            nc_edu_spend = st.slider("Education Spend Ratio", 0.03, 0.20, 0.10, 0.01,
                                     help="% of spending on education (tuition, books etc). RBI data range: 5–15%.")

        with col_d:
            st.subheader("📦 Existing Products")
            st.caption("Check products the customer ALREADY owns.")
            nc_hl  = st.checkbox("🏠 Home Loan")
            nc_cl  = st.checkbox("🚗 Car Loan")
            nc_pl  = st.checkbox("💵 Personal Loan")
            nc_bl  = st.checkbox("🏢 Business Loan")
            nc_gl  = st.checkbox("🪙 Gold Loan")
            nc_el  = st.checkbox("🎓 Education Loan")
            nc_cc  = st.checkbox("💳 Credit Card")
            nc_cc_util = st.slider("  ↳ CC Utilization %", 0, 100, 30,
                                   help="What % of credit limit is currently used? Only matters if Credit Card is checked. High utilization (>50%) hurts CIBIL.")
            nc_ins = st.checkbox("🛡️ Insurance")
            nc_cd  = st.checkbox("📱 Consumer Durable Loan")

        st.divider()
        if st.button("🔍 Get Recommendation", type="primary", use_container_width=True):
            inp = {
                "age": nc_age, "gender": nc_gender, "occupation": nc_occ,
                "employer_type": nc_emp, "education_level": nc_edu,
                "marital_status": nc_mar, "city_tier": nc_cit,
                "monthly_income": float(nc_inc), "cibil_score": int(nc_cibil),
                "repayment_status": nc_rep, "foir": nc_foir, "dti_ratio": nc_dti,
                "bureau_inquiries_6m": nc_bur,
                "writeoff_flag": 1 if nc_wof else 0,
                "default_history_flag": 1 if nc_dfl else 0,
                "savings_rate": nc_sav, "digital_adoption_score": nc_dig,
                "investment_intent_score": nc_inv,
                "online_spend_ratio": nc_onl, "travel_spend_ratio": nc_trv,
                "fuel_spend_ratio": nc_fuel, "education_spend_ratio": nc_edu_spend,
                "tenure_years": nc_tenure,
                "balance_mult": nc_bal_mult,
                "cc_util_pct": nc_cc_util,
                "home_loan_flag": 1 if nc_hl else 0,
                "car_loan_flag":  1 if nc_cl else 0,
                "personal_loan_flag": 1 if nc_pl else 0,
                "business_loan_flag": 1 if nc_bl else 0,
                "gold_loan_flag": 1 if nc_gl else 0,
                "education_loan_flag": 1 if nc_el else 0,
                "credit_card": 1 if nc_cc else 0,
                "insurance": 1 if nc_ins else 0,
                "consumer_durable": 1 if nc_cd else 0,
            }

            with st.spinner("Running ML model..."):
                result, err = predict_for_new_customer(inp)

            if result:
                st.session_state.nc_inp    = inp
                st.session_state.nc_result = result

            if err:
                st.error(err)
            elif result:
                ig_label = get_income_group(float(nc_inc))
                cb_label = get_cibil_bucket(int(nc_cibil))

                # CLV score + tier
                _loan_count = sum([1 if nc_hl else 0, 1 if nc_cl else 0, 1 if nc_pl else 0,
                                   1 if nc_bl else 0, 1 if nc_gl else 0, 1 if nc_el else 0])
                _tenure_pts = min(nc_tenure / 15.0, 1.0) * 30
                _repay_pts  = {"regular": 25, "delayed": 10, "default": 0}.get(nc_rep, 0)
                _cibil_pts  = min(max(max(nc_cibil, 300) - 550, 0) / 350.0, 1.0) * 20
                _prod_pts   = min((_loan_count + (1 if nc_cc else 0) + (1 if nc_ins else 0) + (1 if nc_cd else 0)) / 5.0, 1.0) * 25
                _clv_score  = round(_tenure_pts + _repay_pts + _cibil_pts + _prod_pts, 1)
                _clv_label  = "Super HNI" if _clv_score >= 80 else ("Elite" if _clv_score >= 72 else "Normal")

                st.success("✅ Recommendation Generated")
                st.divider()

                kc1,kc2,kc3,kc4,kc5,kc6 = st.columns(6)
                kc1.metric("Age",     nc_age)
                kc2.metric("CIBIL",   nc_cibil if nc_cibil>0 else "NH", cb_label)
                kc3.metric("Income",  f"₹{nc_inc/1000:.0f}K")
                kc4.metric("FOIR",    f"{nc_foir:.1%}")
                kc5.metric("Income Group", ig_label)
                kc6.metric("CLV Tier", _clv_label, f"Score: {_clv_score}")

                st.divider()
                rc1, rc2 = st.columns(2)

                with rc1:
                    r1  = result["r1_prod"]
                    r1c = result["r1_conf"]
                    r1r = result["r1_risk"]
                    r1a1,r1a2,r1t1,r1t2,r1e1,r1e2 = result["r1_ranges"]
                    rl  = get_risk_label(r1r)
                    re  = RISK_EMOJI.get(rl,"⚪")
                    st.subheader("🥇 Primary Recommendation")
                    st.markdown(f"## {r1.replace('_',' ').title()}")
                    m1,m2 = st.columns(2)
                    cd = "High" if r1c>=70 else ("Medium" if r1c>=40 else "Low ⚠️")
                    m1.metric("Confidence", f"{r1c:.1f}%", cd)
                    m2.metric(f"Bank Risk {re}", f"{r1r:.0f}/100 · {rl}")
                    if r1a2 > 0:
                        st.markdown(f"💰 **Amount:** {fmt(r1a1)} – {fmt(r1a2)}")
                        st.markdown(f"📅 **Tenure:** {r1t1}–{r1t2} months")
                        if r1e2>0: st.markdown(f"📆 **EMI:** {fmt(r1e1)} – {fmt(r1e2)}/mo")

                with rc2:
                    r2  = result["r2_prod"]
                    r2c = result["r2_conf"]
                    r2r = result["r2_risk"]
                    r2a1,r2a2,r2t1,r2t2,r2e1,r2e2 = result["r2_ranges"]
                    rl2 = get_risk_label(r2r)
                    re2 = RISK_EMOJI.get(rl2,"⚪")
                    st.subheader("🥈 Alternative Recommendation")
                    st.markdown(f"## {r2.replace('_',' ').title()}")
                    m3,m4 = st.columns(2)
                    cd2 = "High" if r2c>=70 else ("Medium" if r2c>=40 else "Low")
                    m3.metric("Confidence", f"{r2c:.1f}%", cd2)
                    m4.metric(f"Bank Risk {re2}", f"{r2r:.0f}/100 · {rl2}")
                    if r2a2 > 0:
                        st.markdown(f"💰 **Amount:** {fmt(r2a1)} – {fmt(r2a2)}")
                        st.markdown(f"📅 **Tenure:** {r2t1}–{r2t2} months")
                        if r2e2>0: st.markdown(f"📆 **EMI:** {fmt(r2e1)} – {fmt(r2e2)}/mo")

                st.divider()
                st.subheader("📊 All Product Utility Scores")
                score_rows = []
                for p, s in sorted(result["valid_scores"].items(), key=lambda x: -x[1]):
                    pr = compute_bank_risk_score_simple(
                        p, max(nc_cibil,300), get_cibil_bucket(nc_cibil),
                        nc_rep, 1 if nc_wof else 0, 1 if nc_dfl else 0,
                        nc_bur, nc_foir, nc_dti, nc_age, nc_occ, nc_sav
                    )
                    rl_p = get_risk_label(pr)
                    score_rows.append({
                        "Product": p.replace("_"," ").title(),
                        "Utility Score": f"{s:.4f}",
                        "Bank Risk": f"{pr:.0f}/100",
                        "Risk Label": f"{RISK_EMOJI.get(rl_p,'⚪')} {rl_p}",
                    })
                st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

# ── Tab 4 — Pipeline Breakdown ────────────────────────────────
with tab4:
    st.title("🔍 Pipeline Breakdown")
    st.caption("Step-by-step view of every gate, calculation, and score for the selected customer.")

    _has_existing = bool(customer_id and customer_id in recs_df.index)
    _has_new      = bool(st.session_state.nc_inp and st.session_state.nc_result)

    if not _has_existing and not _has_new:
        st.info("Enter a Customer ID in the sidebar **or** run a New Customer prediction to view the full pipeline.")
    else:
        _options = []
        if _has_existing: _options.append(f"Existing — {customer_id}")
        if _has_new:      _options.append("New Customer (last prediction)")
        _use_existing = True
        if len(_options) > 1:
            _sel = st.radio("Show pipeline for:", _options, horizontal=True)
            _use_existing = _sel.startswith("Existing")
        elif _has_new and not _has_existing:
            _use_existing = False

        # ── Extract unified variables ──────────────────────────────
        if _use_existing:
            _r   = recs_df.loc[customer_id]
            _c   = customers_df.loc[customer_id]
            _a   = asset_df.loc[customer_id]   if customer_id in asset_df.index   else None
            _ci  = cibil_df.loc[customer_id]   if customer_id in cibil_df.index   else None
            _pr  = product_df.loc[customer_id] if customer_id in product_df.index else None
            _age = int(_c["age"]); _occ = str(_c["occupation"])
            _inc = float(_r["monthly_income"]); _ig = str(_r["income_group"])
            _cibil = int(_r["cibil_score"]); _cb = str(_r["cibil_score_bucket"])
            _foir  = float(_r["foir"]); _dti = float(_r["dti_ratio"])
            _dti_b = str(_r["dti_bucket"]); _rep = str(_r["repayment_status"])
            _sav   = float(_r["savings_rate"]); _dig = int(_r["digital_adoption_score"])
            _cit   = int(_r["city_tier"]); _ten = int(_c["tenure_years"])
            _onl   = float(_r["online_spend_ratio"]); _trv = float(_r["travel_spend_ratio"])
            _seg   = str(_c["customer_segment"])
            _clv_score = float(_c["clv_score"]); _clv_tier = str(_c["clv_tier"])
            _wof   = int(_ci["writeoff_flag"])        if _ci is not None else 0
            _dfl   = int(_ci["default_history_flag"]) if _ci is not None else 0
            _bur   = int(_ci["bureau_inquiries_6m"])  if _ci is not None else 0
            _has_hl = int(_a["home_loan_flag"])      if _a is not None else 0
            _has_cl = int(_a["car_loan_flag"])        if _a is not None else 0
            _has_pl = int(_a["personal_loan_flag"])   if _a is not None else 0
            _has_bl = int(_a["business_loan_flag"])   if _a is not None else 0
            _has_gl = int(_a["gold_loan_flag"])       if _a is not None else 0
            _has_el = int(_a["education_loan_flag"])  if _a is not None else 0
            _has_cc  = int(_pr["credit_card"]) if _pr is not None else 0
            _has_ins = int(_pr["insurance"])   if _pr is not None else 0
            _has_cd  = int(_pr["consumer_durable"]) if _pr is not None else 0
            _loan_count = _has_hl + _has_cl + _has_pl + _has_bl + _has_gl + _has_el
            _prod_utility  = {p: float(_r.get(f"score_{p}", 0)) for p in ALL_PRODUCTS}
            _prod_risk     = {p: float(_r.get(f"bank_risk_score_{p}", 0)) for p in ALL_PRODUCTS}
            _proba_by_prod = {}
            _mode_label    = f"Existing Customer — {customer_id}"
        else:
            _inp = st.session_state.nc_inp
            _res = st.session_state.nc_result
            _age = _inp["age"]; _occ = _inp["occupation"]
            _inc = float(_inp["monthly_income"]); _ig = get_income_group(_inc)
            _cibil = int(_inp["cibil_score"]); _cb = get_cibil_bucket(_cibil)
            _foir  = _inp["foir"]; _dti = _inp["dti_ratio"]
            _dti_b = ("severe" if _dti>=5 else "high" if _dti>=3 else "moderate" if _dti>=1.5 else "low")
            _rep   = _inp["repayment_status"]; _sav = _inp["savings_rate"]
            _dig   = _inp["digital_adoption_score"]; _cit = _inp["city_tier"]
            _ten   = _inp.get("tenure_years", 0)
            _onl   = _inp["online_spend_ratio"]; _trv = _inp["travel_spend_ratio"]
            _wof   = _inp.get("writeoff_flag", 0); _dfl = _inp.get("default_history_flag", 0)
            _bur   = _inp.get("bureau_inquiries_6m", 0)
            _has_hl = _inp.get("home_loan_flag",0); _has_cl = _inp.get("car_loan_flag",0)
            _has_pl = _inp.get("personal_loan_flag",0); _has_bl = _inp.get("business_loan_flag",0)
            _has_gl = _inp.get("gold_loan_flag",0); _has_el = _inp.get("education_loan_flag",0)
            _has_cc = _inp.get("credit_card",0); _has_ins = _inp.get("insurance",0)
            _has_cd = _inp.get("consumer_durable",0)
            _loan_count = _has_hl + _has_cl + _has_pl + _has_bl + _has_gl + _has_el
            _cs = max(_cibil, 300)
            _tp = min(_ten/15.0, 1.0)*30
            _rp = {"regular":25,"delayed":10,"default":0}.get(_rep, 0)
            _cp = min(max(_cs-550,0)/350.0, 1.0)*20
            _pp = min((_loan_count + _has_cc + _has_ins + _has_cd)/5.0, 1.0)*25
            _clv_score = round(_tp + _rp + _cp + _pp, 2)
            _clv_tier  = ("super_hni" if _clv_score>=80 else "elite" if _clv_score>=72 else "normal")
            _seg = ("youth" if _age<30 else
                    "elite" if (_ig=="high" and _age<45) else
                    "hni"   if _ig=="high" else
                    "mass_affluent" if _ig=="upper_mid" else
                    "retail" if _ig=="mid" else "mass")
            _prod_utility  = _res.get("valid_scores", {})
            _prod_risk     = {p: compute_bank_risk_score_simple(
                                 p, max(_cibil,300), _cb, _rep, _wof, _dfl,
                                 _bur, _foir, _dti, _age, _occ, _sav)
                             for p in ALL_PRODUCTS}
            _proba_by_prod = _res.get("proba_by_product", {})
            _mode_label    = "New Customer (last prediction)"

        st.markdown(f"**Source:** {_mode_label}")
        st.divider()

        # ── STEP 1: PROFILE ─────────────────────────────────────────
        st.markdown("### Step 1 — Customer Profile")
        _pc = st.columns(7)
        _pc[0].metric("Age", _age)
        _pc[1].metric("Income", f"₹{_inc/1000:.0f}K", _ig.replace("_"," ").title())
        _pc[2].metric("CIBIL", _cibil if _cibil>0 else "NH", _cb.replace("_"," ").title())
        _pc[3].metric("FOIR", f"{_foir:.1%}")
        _pc[4].metric("DTI", f"{_dti:.1f}×", _dti_b.title())
        _pc[5].metric("Tenure", f"{_ten} yrs")
        _pc[6].metric("City", f"Tier {_cit}")
        _pc2 = st.columns(7)
        _pc2[0].metric("Occupation", _occ.replace("_"," ").title())
        _pc2[1].metric("Repayment",  _rep.title())
        _pc2[2].metric("Savings",    f"{_sav:.0%}")
        _pc2[3].metric("Digital",    f"{_dig}/3")
        _pc2[4].metric("Online Spd", f"{_onl:.0%}")
        _pc2[5].metric("Travel Spd", f"{_trv:.0%}")
        _pc2[6].metric("Segment",    _seg.replace("_"," ").title())
        _owned = [lbl for lbl,v in [
            ("Home Loan",_has_hl),("Car Loan",_has_cl),("Personal Loan",_has_pl),
            ("Business Loan",_has_bl),("Gold Loan",_has_gl),("Education Loan",_has_el),
            ("Credit Card",_has_cc),("Insurance",_has_ins),("Consumer Durable",_has_cd)] if v]
        st.caption(f"**Products already owned:** {', '.join(_owned) if _owned else 'None'}")

        # ── STEP 2: HARD BLOCKERS ────────────────────────────────────
        st.divider()
        st.markdown("### Step 2 — Hard Blockers")
        _b1, _b2, _b3 = st.columns(3)
        with _b1:
            if _dti >= 5:
                st.error(f"**DTI Severe — {_dti:.1f}× ≥ 5.0**  \nBlocks all new credit products")
            else:
                st.success(f"**DTI OK — {_dti:.1f}× ({_dti_b})**  \nNo block")
        with _b2:
            if _wof:
                st.error("**Writeoff / NPA on record**  \nBlocks all credit except insurance & gold loan")
            else:
                st.success("**No Writeoff / NPA**  \nNo block")
        with _b3:
            if _dfl:
                st.error("**Prior Default History**  \nBlocks all credit except insurance & gold loan")
            else:
                st.success("**No Default History**  \nNo block")

        # ── STEP 3: CLV SCORE ────────────────────────────────────────
        st.divider()
        st.markdown("### Step 3 — CLV Score Calculation")
        _cs2 = max(_cibil, 300)
        if _use_existing:
            _tp2 = min(_ten/15.0,1.0)*30
            _rp2 = {"regular":25,"delayed":10,"default":0}.get(_rep,0)
            _cp2 = min(max(_cs2-550,0)/350.0,1.0)*20
            _prod_count_clv = _loan_count + _has_cc + _has_ins + _has_cd
            _pp2 = min(_prod_count_clv/5.0,1.0)*25
        else:
            _tp2, _rp2, _cp2, _pp2 = _tp, _rp, _cp, _pp
            _prod_count_clv = _loan_count + _has_cc + _has_ins + _has_cd
        _clv_calc = round(_tp2 + _rp2 + _cp2 + _pp2, 2)
        _tier_lbl  = {"super_hni":"Super HNI 🔵","elite":"Elite 🟣","normal":"Normal ⚪"}.get(
                      _clv_tier.lower(), _clv_tier.upper())

        _clv_col1, _clv_col2 = st.columns([3, 1])
        with _clv_col1:
            st.markdown(f"""
| Component | Calculation | Points |
|---|---|---|
| Tenure pts | min({_ten} / 15, 1.0) × 30 | **{_tp2:.2f}** |
| Repayment pts | `{_rep}` → {_rp2} | **{_rp2:.1f}** |
| CIBIL pts | (max({_cibil},300) − 550) / 350 × 20 | **{_cp2:.2f}** |
| Product pts | {_prod_count_clv} products (loans+CC+ins+CD) / 5 × 25 | **{_pp2:.2f}** |
| **CLV Score** | sum | **{_clv_calc} / 100** |
""")
        with _clv_col2:
            st.metric("CLV Score", f"{_clv_score:.1f}", _tier_lbl)
            st.caption("≥ 80 = Super HNI  \n≥ 72 = Elite  \n< 72 = Normal")

        # ── STEP 4: RISK CATEGORY ────────────────────────────────────
        st.divider()
        st.markdown("### Step 4 — Risk Category Derivation")
        _rc = st.columns(4)
        _checks = [
            ("Writeoff flag?",          bool(_wof),
             f"{'YES — triggers HIGH' if _wof else 'No'}"),
            ("Default history?",        bool(_dfl),
             f"{'YES — triggers HIGH' if _dfl else 'No'}"),
            (f"CIBIL {_cs2} < 650?",    _cs2 < 650,
             f"{'YES — triggers HIGH' if _cs2<650 else 'No — CIBIL is ' + str(_cs2)}"),
            (f"CIBIL ≥ 750 + regular + bureau < 3?",
             _cs2>=750 and _rep=="regular" and _bur<3,
             f"CIBIL={_cs2}, rep={_rep}, bureau={_bur}"),
        ]
        for col, (label, hit, detail) in zip(_rc, _checks):
            if label.startswith("CIBIL ≥"):
                if hit:
                    col.success(f"**{label}**  \n{detail}  \n→ **LOW risk**")
                else:
                    col.info(f"**{label}**  \n{detail}  \nnot met")
            elif hit:
                col.error(f"**{label}**  \n{detail}")
            else:
                col.success(f"**{label}**  \n{detail}")

        if _wof or _dfl or _cs2 < 650 or _rep == "default":
            _risk_final = "HIGH (ord=2)"
            _risk_color = "error"
            _risk_why   = ("Writeoff flag" if _wof else "Default history" if _dfl else
                           f"CIBIL {_cs2} < 650" if _cs2<650 else "Repayment = default")
        elif _cs2>=750 and _rep=="regular" and _bur<3:
            _risk_final = "LOW (ord=0)"
            _risk_color = "success"
            _risk_why   = f"CIBIL {_cs2} ≥ 750, regular repayment, bureau inquiries {_bur} < 3"
        else:
            _risk_final = "MEDIUM (ord=1)"
            _risk_color = "warning"
            _risk_why   = f"No high-risk flags but CIBIL {_cs2} < 750 or bureau inquiries ≥ 3"
        getattr(st, _risk_color)(f"**→ Risk Category: {_risk_final}** — {_risk_why}")

        # ── STEP 5: ELIGIBILITY GATES ────────────────────────────────
        st.divider()
        st.markdown("### Step 5 — Eligibility Gates (9 Products)")
        _gate_rows = []
        for _prod in ALL_PRODUCTS:
            _g = AGE_GATES.get(_prod, {})
            _age_ok = ((_g.get("max_age",999) >= _age) and (_g.get("min_age",0) <= _age))
            _age_note = ("✅" if not _g else
                         f"✅ {_age}≤{_g['max_age']}" if "max_age" in _g and _age<=_g["max_age"] else
                         f"✅ {_age}≥{_g['min_age']}" if "min_age" in _g and _age>=_g["min_age"] else
                         f"❌ age {_age} out of {_g.get('min_age','?')}–{_g.get('max_age','?')}")
            if not _age_ok:
                _age_note = f"❌ {_age} not in {_g.get('min_age','?')}–{_g.get('max_age','?')}"
            _occ_ok   = not (_occ=="student" and _prod in STUDENT_BLOCKED)
            _occ_note = "✅" if _occ_ok else "❌ student"
            _dti_ok   = not (_dti>=5 and _prod!="insurance")
            _dti_note = f"✅ {_dti:.1f}×" if _dti_ok else f"❌ severe {_dti:.1f}×"
            _foir_ok  = is_affordable(_prod, _ig, _inc, _foir)
            _foir_note = "N/A ✅" if (_prod not in LOAN_AMOUNT_RANGES or _prod=="gold_loan") else \
                         (f"✅ {_foir:.0%}" if _foir_ok else f"❌ {_foir:.0%} too high")
            if _prod not in LOAN_AMOUNT_RANGES or _prod=="gold_loan":
                _foir_ok = True
            _biz_ok = True
            if _prod == "business_loan":
                if _ig in ("low","lower_mid") or (_ig=="mid" and _foir>=0.35) or _foir>=0.35:
                    _biz_ok = False
            _wof_ok = not (_wof and _prod not in ["insurance"])
            _dfl_ok = not (_dfl and _prod not in ["insurance","gold_loan"])
            _eligible = _age_ok and _occ_ok and _dti_ok and _foir_ok and _biz_ok and _wof_ok and _dfl_ok
            _gate_rows.append({
                "Product"    : _prod.replace("_"," ").title(),
                "Age"        : _age_note,
                "Occupation" : _occ_note,
                "DTI"        : _dti_note,
                "FOIR/Afford": _foir_note,
                "Writeoff"   : ("✅" if _wof_ok else "❌ NPA"),
                "Default"    : ("✅" if _dfl_ok else "❌ blocked"),
                "Result"     : "✅ Eligible" if _eligible else "❌ Blocked",
            })
        st.dataframe(pd.DataFrame(_gate_rows), use_container_width=True, hide_index=True)

        # ── STEP 6: BANK RISK SCORE BREAKDOWN ───────────────────────
        st.divider()
        st.markdown("### Step 6 — Bank Risk Score per Product")
        st.caption("Formula: base + CIBIL adj + repayment adj + FOIR adj + DTI adj + bureau adj")
        _base_map = {"home_loan":20,"car_loan":22,"personal_loan":38,"education_loan":25,
                     "gold_loan":12,"business_loan":42,"credit_card":28,"insurance":5,"consumer_durable":30}
        _cadj = {"excellent":-18,"good":-8,"risky":+15,"high_risk":+30,"no_history":+12}.get(_cb,0)
        _radj = {"regular":-8,"delayed":+10,"default":+25}.get(_rep,0)
        _fadj = (+20 if _foir>=0.50 else +12 if _foir>=0.40 else +5 if _foir>=0.30 else +2 if _foir>=0.20 else -3)
        _dadj = (+18 if _dti>=5 else +10 if _dti>=3 else +4 if _dti>=1.5 else -2)
        _badj = min(_bur*4, 20)
        _risk_adj_str = (f"CIBIL({_cb})={_cadj:+d} | Repay({_rep})={_radj:+d} | "
                         f"FOIR({_foir:.0%})={_fadj:+d} | DTI({_dti:.1f}×)={_dadj:+d} | "
                         f"Bureau({_bur})={_badj:+d}")
        st.caption(f"Shared adjustments: {_risk_adj_str}")
        _risk_rows = []
        for _prod in ALL_PRODUCTS:
            _base = _base_map.get(_prod, 25)
            _r    = _prod_risk.get(_prod, 0)
            _rl   = get_risk_label(_r)
            _risk_rows.append({
                "Product"   : _prod.replace("_"," ").title(),
                "Base"      : _base,
                "Adjustments": _risk_adj_str.split("|")[0].strip()[:20]+"…",
                "Final Score": f"{_r:.0f}/100",
                "Label"     : f"{RISK_EMOJI.get(_rl,'⚪')} {_rl}",
            })
        st.dataframe(pd.DataFrame(_risk_rows), use_container_width=True, hide_index=True)

        # ── STEP 7: UTILITY SCORES ───────────────────────────────────
        st.divider()
        st.markdown("### Step 7 — Product Utility Scores")
        st.caption("Utility = ML Confidence × 0.45 + (Profit/100) × 0.35 − (BankRisk/100) × 0.20  ×  Saturation multiplier")
        _util_rows = []
        for _prod in ALL_PRODUCTS:
            _util = _prod_utility.get(_prod, 0)
            _r2   = _prod_risk.get(_prod, 0)
            _rl2  = get_risk_label(_r2)
            _conf_val  = _proba_by_prod.get(_prod, None)
            _conf_str  = f"{_conf_val*100:.1f}%" if _conf_val is not None else ("from CSV" if _use_existing else "—")
            _profit    = PRODUCT_PROFIT.get(_prod, 0)
            _sat       = PRODUCT_SATURATION.get(_prod, 1.0)
            _elig_flag = passes_all_gates(_prod, _age, _occ, _ig, _inc, _foir) and \
                         not (_dti>=5 and _prod!="insurance") and \
                         not (_wof and _prod not in ["insurance"]) and \
                         not (_dfl and _prod not in ["insurance","gold_loan"])
            _util_rows.append({
                "Product"     : _prod.replace("_"," ").title(),
                "ML Conf"     : _conf_str,
                "Profit/100"  : f"{_profit/100:.2f}",
                "Bank Risk"   : f"{_r2:.0f}/100",
                "Saturation"  : f"×{_sat:.2f}",
                "Utility"     : f"{_util:.4f}" if _util > 0 else "0 (blocked/ineligible)",
                "Eligible"    : "✅" if _elig_flag else "❌",
            })
        _util_rows.sort(key=lambda x: float(x["Utility"].split()[0]) if x["Eligible"]=="✅" else -1, reverse=True)
        st.dataframe(pd.DataFrame(_util_rows), use_container_width=True, hide_index=True)

        # ── STEP 8: FINAL RANKING ────────────────────────────────────
        st.divider()
        st.markdown("### Step 8 — Final Ranking")
        _ranked = sorted(
            [(p, _prod_utility.get(p,0)) for p in ALL_PRODUCTS if _prod_utility.get(p,0) > 0],
            key=lambda x: -x[1]
        )
        if _ranked:
            _medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣"]
            _rank_cols = st.columns(min(len(_ranked), 4))
            for _i, (_p, _s) in enumerate(_ranked):
                _rsk = _prod_risk.get(_p, 0)
                _rl3 = get_risk_label(_rsk)
                _medal = _medals[_i] if _i < len(_medals) else f"{_i+1}."
                if _i < 4:
                    with _rank_cols[_i]:
                        st.metric(f"{_medal} {_p.replace('_',' ').title()}", f"{_s:.4f}", f"Risk {_rsk:.0f}/100 · {_rl3}")
            if len(_ranked) > 4:
                for _i, (_p, _s) in enumerate(_ranked[4:], 4):
                    _rsk = _prod_risk.get(_p, 0)
                    _rl3 = get_risk_label(_rsk)
                    _medal = _medals[_i] if _i < len(_medals) else f"{_i+1}."
                    st.markdown(f"{_medal} **{_p.replace('_',' ').title()}** — utility `{_s:.4f}` | risk `{_rsk:.0f}/100` ({_rl3})")
        else:
            st.warning("No eligible products with utility score > 0.")

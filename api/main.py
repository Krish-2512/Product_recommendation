from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(
    title="ICICI Bank Recommendation API",
    description="ML-powered product recommendation engine for ICICI Bank customers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data once at startup ─────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")

recs_df      = pd.read_csv(os.path.join(BASE, "recommendations_output_v5_1.csv")).set_index("customer_id")
shap_df      = pd.read_csv(os.path.join(BASE, "shap_top_features.csv")).set_index("customer_id")
customers_df = pd.read_csv(os.path.join(BASE, "customer_master.csv")).set_index("customer_id")
perf_df      = pd.read_csv(os.path.join(BASE, "product_performance_monthly.csv"))

ALL_PRODUCTS = [
    "home_loan", "car_loan", "education_loan", "personal_loan",
    "business_loan", "gold_loan", "credit_card", "insurance", "consumer_durable"
]

def fmt(v: float) -> str:
    if v >= 100_000: return f"₹{v/100_000:.1f}L"
    if v >= 1_000:   return f"₹{v/1_000:.0f}K"
    return f"₹{v:.0f}"

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "ICICI Bank Recommendation API",
        "version": "1.0.0",
        "total_customers": len(recs_df)
    }

@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    customer_id = customer_id.upper()
    if customer_id not in customers_df.index:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    row = customers_df.loc[customer_id]
    return {
        "customer_id"      : customer_id,
        "age"              : int(row["age"]),
        "gender"           : row["gender"],
        "occupation"       : row["occupation"],
        "income_group"     : row["income_group"],
        "monthly_income"   : float(row["monthly_income"]),
        "customer_segment" : row["customer_segment"],
        "city"             : row["city"],
        "state"            : row["state"],
        "clv_tier"         : row["clv_tier"],
        "clv_score"        : float(row["clv_score"]),
        "tenure_years"     : int(row["tenure_years"]),
        "risk_category"    : row["risk_category"],
    }

@app.get("/recommend/{customer_id}")
def get_recommendation(customer_id: str):
    customer_id = customer_id.upper()
    if customer_id not in recs_df.index:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    row = recs_df.loc[customer_id]
    r1  = row["ml_rank1_recommendation"]
    r2  = row["ml_rank2_recommendation"]

    return {
        "customer_id": customer_id,
        "rank1": {
            "product"         : r1,
            "confidence_pct"  : round(float(row["rank1_confidence_pct"]), 2),
            "amount_range"    : f"{fmt(row['rank1_amount_min'])} – {fmt(row['rank1_amount_max'])}" if row["rank1_amount_max"] > 0 else "N/A",
            "tenure_range"    : f"{int(row['rank1_tenure_min_months'])}–{int(row['rank1_tenure_max_months'])} months" if row["rank1_tenure_max_months"] > 0 else "N/A",
            "emi_range"       : f"{fmt(row['rank1_emi_min'])} – {fmt(row['rank1_emi_max'])}/mo" if row["rank1_emi_max"] > 0 else "N/A",
            "bank_risk_score" : float(row[f"bank_risk_score_{r1}"]),
            "bank_risk_label" : row[f"bank_risk_label_{r1}"],
            "utility_score"   : round(float(row["utility_score_rank1"]), 4),
        },
        "rank2": {
            "product"        : r2,
            "confidence_pct" : round(float(row["rank2_confidence_pct"]), 2),
            "amount_range"   : f"{fmt(row['rank2_amount_min'])} – {fmt(row['rank2_amount_max'])}" if row["rank2_amount_max"] > 0 else "N/A",
            "tenure_range"   : f"{int(row['rank2_tenure_min_months'])}–{int(row['rank2_tenure_max_months'])} months" if row["rank2_tenure_max_months"] > 0 else "N/A",
            "emi_range"      : f"{fmt(row['rank2_emi_min'])} – {fmt(row['rank2_emi_max'])}/mo" if row["rank2_emi_max"] > 0 else "N/A",
        },
        "credit_profile": {
            "cibil_score"      : int(row["cibil_score"]),
            "cibil_bucket"     : row["cibil_score_bucket"],
            "income_group"     : row["income_group"],
            "occupation"       : row["occupation"],
            "foir"             : round(float(row["foir"]), 4),
            "dti_ratio"        : round(float(row["dti_ratio"]), 4),
            "dti_bucket"       : row["dti_bucket"],
            "repayment_status" : row["repayment_status"],
        }
    }

@app.get("/explain/{customer_id}")
def get_explanation(customer_id: str):
    customer_id = customer_id.upper()
    if customer_id not in shap_df.index:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    row = shap_df.loc[customer_id]
    return {
        "customer_id"        : customer_id,
        "recommended_product": row["recommended_product"],
        "confidence_pct"     : float(row["confidence_pct"]),
        "clv_tier"           : row["clv_tier"],
        "top_reasons": [
            {"rank": 1, "feature": row["top1_feature"], "shap_impact": round(float(row["top1_shap"]), 4), "customer_value": round(float(row["top1_value"]), 4)},
            {"rank": 2, "feature": row["top2_feature"], "shap_impact": round(float(row["top2_shap"]), 4), "customer_value": round(float(row["top2_value"]), 4)},
            {"rank": 3, "feature": row["top3_feature"], "shap_impact": round(float(row["top3_shap"]), 4), "customer_value": round(float(row["top3_value"]), 4)},
        ],
        "top_blocker": {
            "feature"     : row["top_blocker_feature"],
            "shap_impact" : round(float(row["top_blocker_shap"]), 4),
        }
    }

@app.get("/risk/{customer_id}")
def get_risk_scores(customer_id: str):
    customer_id = customer_id.upper()
    if customer_id not in recs_df.index:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    row = recs_df.loc[customer_id]
    return {
        "customer_id": customer_id,
        "risk_scores": {
            p: {
                "score": float(row[f"bank_risk_score_{p}"]),
                "label": row[f"bank_risk_label_{p}"]
            }
            for p in ALL_PRODUCTS
        }
    }

@app.get("/products")
def get_products():
    latest = perf_df[perf_df["month"] == perf_df["month"].max()]
    perf   = latest.set_index("product")[["default_rate_pct", "profit_multiplier", "net_adjustment"]].to_dict("index")
    return {
        "products"   : ALL_PRODUCTS,
        "performance": perf,
    }

@app.get("/stats")
def get_stats():
    return {
        "total_customers"      : len(recs_df),
        "recommendation_dist"  : recs_df["ml_rank1_recommendation"].value_counts().to_dict(),
        "cibil_distribution": {
            "excellent_gte750"  : int((recs_df["cibil_score"] >= 750).sum()),
            "good_700_749"      : int(((recs_df["cibil_score"] >= 700) & (recs_df["cibil_score"] < 750)).sum()),
            "risky_650_699"     : int(((recs_df["cibil_score"] >= 650) & (recs_df["cibil_score"] < 700)).sum()),
            "high_risk_lt650"   : int((recs_df["cibil_score"] < 650).sum()),
        },
        "avg_rank1_confidence" : round(float(recs_df["rank1_confidence_pct"].mean()), 2),
        "rank1_match_rate"     : round(float(recs_df["rank1_match"].mean()), 4),
    }

"""
D2C Customer Churn Scoring API
==============================
FastAPI service that loads a trained LightGBM churn model and exposes
three endpoints for the internal CRM retention team.

Endpoints
---------
GET  /health          — liveness check
POST /predict         — single customer churn score
POST /batch_predict   — multiple customers in one call
"""

import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Model loading — done once at startup via lifespan
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
_bundle: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle at startup; release at shutdown."""
    global _bundle
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"model.pkl not found at '{MODEL_PATH}'. "
            "Set the MODEL_PATH env variable or place model.pkl in the working directory."
        )
    _bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "cat_encoders", "cat_features",
                     "num_features", "all_features", "best_threshold"}
    missing = required_keys - set(_bundle.keys())
    if missing:
        raise RuntimeError(f"model.pkl is missing required keys: {missing}")
    print(f"[startup] Model loaded — threshold={_bundle['best_threshold']}, "
          f"features={len(_bundle['all_features'])}")
    yield
    _bundle.clear()
    print("[shutdown] Model bundle released.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="D2C Churn Scoring API",
    description=(
        "Internal API for churn-risk scoring. "
        "Returns churn probability, predicted class, risk level, "
        "and a plain-language risk explanation for retention team use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas — input validation
# ---------------------------------------------------------------------------

# Valid categorical values (mirrors LabelEncoder classes from training)
VALID_CITY_TIER           = {"Tier 1", "Tier 2", "Tier 3"}
VALID_AGE_GROUP           = {"18-24", "25-34", "35-44", "45+"}
VALID_ACQUISITION_CHANNEL = {"Google Search", "Influencer", "Instagram",
                              "Marketplace", "Organic", "Referral"}
VALID_LOYALTY_TIER        = {"Gold", "Not_Enrolled", "Platinum", "Silver"}
VALID_PREFERRED_CATEGORY  = {"Baby Care", "Fragrance", "Hair Care",
                              "Makeup", "Skin Care", "Wellness"}
VALID_MARKETING_CONSENT   = {"No", "Yes"}


class CustomerFeatures(BaseModel):
    """
    Feature payload for a single customer.

    The 4 engineered features (engagement_rate, email_click_rate,
    spend_per_order, recency_x_frequency) are computed automatically
    from the base inputs — callers do NOT need to supply them.
    """

    # ── Identifiers (optional, echoed back in response) ──────────────────
    customer_id: Optional[str] = Field(None, description="Optional customer ID for traceability")

    # ── Categorical features ──────────────────────────────────────────────
    city_tier: str = Field(..., description="Customer city tier: 'Tier 1', 'Tier 2', or 'Tier 3'")
    age_group: str = Field(..., description="Age bracket: '18-24', '25-34', '35-44', '45+'")
    acquisition_channel: str = Field(..., description="Marketing acquisition channel")
    loyalty_tier: str = Field(
        "Not_Enrolled",
        description="Loyalty programme tier. Use 'Not_Enrolled' if not in programme."
    )
    preferred_category: str = Field(..., description="Customer's preferred product category")
    marketing_consent: str = Field(..., description="'Yes' or 'No'")

    # ── RFM features ─────────────────────────────────────────────────────
    recency_days: float = Field(..., ge=0, description="Days since last order")
    frequency_180d: float = Field(..., ge=0, description="Number of orders in last 180 days")
    monetary_180d: float = Field(..., ge=0, description="Total spend (₹) in last 180 days")

    # ── Order behaviour ───────────────────────────────────────────────────
    return_rate_180d: float = Field(..., ge=0.0, le=1.0, description="Proportion of orders returned (0–1)")
    avg_discount_pct_180d: float = Field(..., ge=0.0, le=1.0, description="Average discount fraction (0–1)")
    avg_rating_180d: float = Field(..., ge=0.0, le=5.0, description="Average order rating (0–5; 0 if no ratings)")
    category_diversity_180d: float = Field(..., ge=0, description="Number of distinct categories purchased")

    # ── Support signals ───────────────────────────────────────────────────
    ticket_count_90d: float = Field(..., ge=0, description="Support tickets raised in last 90 days")
    negative_ticket_rate_90d: float = Field(..., ge=0.0, le=1.0, description="Fraction of tickets with negative sentiment")
    avg_resolution_hours_90d: float = Field(..., ge=0, description="Average ticket resolution time (hours)")

    # ── Profile ───────────────────────────────────────────────────────────
    days_since_signup: float = Field(..., ge=0, description="Days since account creation")

    # ── Web / app activity ────────────────────────────────────────────────
    sessions_30d: float = Field(..., ge=0, description="Web/app sessions in last 30 days")
    product_views_30d: float = Field(..., ge=0, description="Product pages viewed in last 30 days")
    cart_adds_30d: float = Field(..., ge=0, description="Items added to cart in last 30 days")
    wishlist_adds_30d: float = Field(..., ge=0, description="Items added to wishlist in last 30 days")
    abandoned_carts_30d: float = Field(..., ge=0, description="Cart sessions ending without purchase")
    email_opens_30d: float = Field(..., ge=0, description="Marketing emails opened in last 30 days")
    campaign_clicks_30d: float = Field(..., ge=0, description="Campaign link clicks in last 30 days")
    last_visit_days_ago: float = Field(..., ge=0, description="Days since most recent site/app visit")

    # ── Categorical validators ────────────────────────────────────────────
    @field_validator("city_tier")
    @classmethod
    def validate_city_tier(cls, v):
        if v not in VALID_CITY_TIER:
            raise ValueError(f"city_tier must be one of {sorted(VALID_CITY_TIER)}")
        return v

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v):
        if v not in VALID_AGE_GROUP:
            raise ValueError(f"age_group must be one of {sorted(VALID_AGE_GROUP)}")
        return v

    @field_validator("acquisition_channel")
    @classmethod
    def validate_acquisition_channel(cls, v):
        if v not in VALID_ACQUISITION_CHANNEL:
            raise ValueError(f"acquisition_channel must be one of {sorted(VALID_ACQUISITION_CHANNEL)}")
        return v

    @field_validator("loyalty_tier")
    @classmethod
    def validate_loyalty_tier(cls, v):
        if v not in VALID_LOYALTY_TIER:
            raise ValueError(f"loyalty_tier must be one of {sorted(VALID_LOYALTY_TIER)}")
        return v

    @field_validator("preferred_category")
    @classmethod
    def validate_preferred_category(cls, v):
        if v not in VALID_PREFERRED_CATEGORY:
            raise ValueError(f"preferred_category must be one of {sorted(VALID_PREFERRED_CATEGORY)}")
        return v

    @field_validator("marketing_consent")
    @classmethod
    def validate_marketing_consent(cls, v):
        if v not in VALID_MARKETING_CONSENT:
            raise ValueError(f"marketing_consent must be 'Yes' or 'No'")
        return v

    @model_validator(mode="after")
    def recency_vs_signup(self):
        """Sanity check: recency cannot exceed days_since_signup."""
        if self.recency_days > self.days_since_signup + 1:
            raise ValueError(
                f"recency_days ({self.recency_days}) cannot exceed days_since_signup "
                f"({self.days_since_signup}). The customer cannot have last ordered "
                f"before they signed up."
            )
        return self


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of customer feature payloads (1–500 per batch)"
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    threshold: float
    feature_count: int


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def _encode_and_predict(customers: List[CustomerFeatures]) -> List[PredictionResponse]:
    """
    Encode categorical features, compute engineered features,
    run model inference, and build response objects.
    """
    model        = _bundle["model"]
    cat_encoders = _bundle["cat_encoders"]
    cat_features = _bundle["cat_features"]
    all_features = _bundle["all_features"]
    threshold    = _bundle["best_threshold"]

    rows = []
    for c in customers:
        row = {
            "city_tier":                c.city_tier,
            "age_group":                c.age_group,
            "acquisition_channel":      c.acquisition_channel,
            "loyalty_tier":             c.loyalty_tier,
            "preferred_category":       c.preferred_category,
            "marketing_consent":        c.marketing_consent,
            "recency_days":             c.recency_days,
            "frequency_180d":           c.frequency_180d,
            "monetary_180d":            c.monetary_180d,
            "return_rate_180d":         c.return_rate_180d,
            "avg_discount_pct_180d":    c.avg_discount_pct_180d,
            "avg_rating_180d":          c.avg_rating_180d,
            "category_diversity_180d":  c.category_diversity_180d,
            "ticket_count_90d":         c.ticket_count_90d,
            "negative_ticket_rate_90d": c.negative_ticket_rate_90d,
            "avg_resolution_hours_90d": c.avg_resolution_hours_90d,
            "days_since_signup":        c.days_since_signup,
            "sessions_30d":             c.sessions_30d,
            "product_views_30d":        c.product_views_30d,
            "cart_adds_30d":            c.cart_adds_30d,
            "wishlist_adds_30d":        c.wishlist_adds_30d,
            "abandoned_carts_30d":      c.abandoned_carts_30d,
            "email_opens_30d":          c.email_opens_30d,
            "campaign_clicks_30d":      c.campaign_clicks_30d,
            "last_visit_days_ago":      c.last_visit_days_ago,
            # Engineered features — computed here, not required from caller
            "engagement_rate":     c.cart_adds_30d / (c.product_views_30d + 1),
            "email_click_rate":    c.campaign_clicks_30d / (c.email_opens_30d + 1),
            "spend_per_order":     c.monetary_180d / (c.frequency_180d + 1),
            "recency_x_frequency": math.log1p(c.recency_days) * (c.frequency_180d + 1),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Label-encode categoricals (use -1 for unseen values as a safe fallback)
    for col in cat_features:
        le = cat_encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
        )

    # Predict
    probas = model.predict_proba(df[all_features])[:, 1]

    results = []
    for c, proba in zip(customers, probas):
        pred_class  = int(proba >= threshold)
        risk_level  = _risk_level(proba)
        explanation = _risk_explanation(c, proba)
        results.append(PredictionResponse(
            customer_id       = c.customer_id,
            churn_probability = round(float(proba), 4),
            predicted_class   = pred_class,
            risk_level        = risk_level,
            risk_explanation  = explanation,
            threshold_used    = threshold,
        ))
    return results


def _risk_level(proba: float) -> str:
    """Bucket probability into a human-readable risk tier."""
    if proba < 0.35:
        return "low"
    elif proba < 0.60:
        return "medium"
    elif proba < 0.80:
        return "high"
    else:
        return "critical"


def _risk_explanation(c: CustomerFeatures, proba: float) -> str:
    """
    Generate a plain-language explanation from the customer's top risk signals.
    Logic mirrors the SHAP feature importance from training:
      1. recency_days        (weight ~0.36)
      2. last_visit_days_ago (weight ~0.26)
      3. monetary_180d       (weight ~0.17)
      4. recency_x_frequency (weight ~0.09)
      5. sessions_30d        (weight ~0.07)
    """
    signals = []

    # Recency
    if c.recency_days > 150:
        signals.append(f"very high recency ({int(c.recency_days)} days since last order)")
    elif c.recency_days > 90:
        signals.append(f"high recency ({int(c.recency_days)} days since last order)")
    elif c.recency_days > 60:
        signals.append(f"moderate recency ({int(c.recency_days)} days since last order)")

    # Last visit / web disengagement
    if c.last_visit_days_ago > 30:
        signals.append(f"no website/app visit in {int(c.last_visit_days_ago)} days")
    elif c.last_visit_days_ago > 14:
        signals.append(f"low recent web activity (last visit {int(c.last_visit_days_ago)} days ago)")

    # Monetary value (absence of spend)
    if c.monetary_180d == 0:
        signals.append("no purchases in the last 180 days")
    elif c.monetary_180d < 300:
        signals.append(f"very low 180-day spend (₹{c.monetary_180d:.0f})")

    # Session engagement
    if c.sessions_30d == 0:
        signals.append("zero web/app sessions in the last 30 days")
    elif c.sessions_30d <= 1:
        signals.append(f"minimal web engagement ({int(c.sessions_30d)} session in 30 days)")

    # Support issues
    if c.ticket_count_90d >= 2 and c.negative_ticket_rate_90d >= 0.5:
        signals.append(
            f"{int(c.ticket_count_90d)} support tickets with "
            f"{c.negative_ticket_rate_90d:.0%} negative sentiment"
        )
    elif c.ticket_count_90d >= 1 and c.negative_ticket_rate_90d == 1.0:
        signals.append("recent support ticket with fully negative sentiment")

    # Return rate
    if c.return_rate_180d >= 0.5:
        signals.append(f"high return rate ({c.return_rate_180d:.0%} of orders returned)")

    # Low frequency
    if c.frequency_180d == 0:
        signals.append("no orders in the last 180 days")
    elif c.frequency_180d == 1 and c.recency_days > 60:
        signals.append("only one order in 180 days")

    # Fallback for low-risk predictions
    if not signals:
        if proba < 0.35:
            return (
                "Customer shows healthy engagement — recent purchases, active web sessions, "
                "and positive support history indicate low churn risk."
            )
        else:
            return (
                "Combination of moderately high recency and reduced engagement "
                "signals elevated churn risk."
            )

    # Compose explanation
    signal_str = "; ".join(signals[:3])   # cap at 3 for readability
    if proba >= 0.80:
        prefix = "Critical churn risk: "
    elif proba >= 0.60:
        prefix = "High churn risk: "
    elif proba >= 0.35:
        prefix = "Moderate churn risk: "
    else:
        prefix = "Low churn risk despite: "

    return prefix + signal_str + "."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Liveness check. Returns API status and model metadata.
    Use this endpoint to confirm the service is running and the model is loaded.
    """
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status        = "ok",
        model         = "LightGBM Churn Classifier v1.0",
        version       = "1.0.0",
        threshold     = _bundle["best_threshold"],
        feature_count = len(_bundle["all_features"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Score a single customer for churn risk.

    **Input:** Customer feature payload (25 raw features).  
    The 4 engineered features are computed automatically — do not include them.

    **Output:** Churn probability, predicted class (0/1), risk level
    (low / medium / high / critical), and a plain-language risk explanation.

    **Threshold:** 0.35 — optimised to maximise recall (catch churners)
    while keeping precision above 67%.
    """
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = _encode_and_predict([customer])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
def batch_predict(request: BatchPredictRequest):
    """
    Score multiple customers in a single call (up to 500 per request).

    Returns individual predictions for each customer plus a summary
    count of high-risk customers and total processing time.
    """
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    try:
        results = _encode_and_predict(request.customers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    high_risk  = sum(1 for r in results if r.risk_level in ("high", "critical"))

    return BatchPredictionResponse(
        predictions         = results,
        total_customers     = len(results),
        high_risk_count     = high_risk,
        processing_time_ms  = round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Global error handler — return JSON instead of HTML for unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# D2C Customer Churn — Part 4: FastAPI Churn Scoring Service

## Student Details

Student Name: Mohammed Rashiku B C  
Student ID: iitp_aiml_25061023

## Overview

This repository contains Part 4 of the D2C Customer Churn Capstone.  
A trained LightGBM churn model (from Part 3) is wrapped in a FastAPI service
that the internal CRM team can call to score customers for churn risk before
launching a retention campaign.

**Model:** LightGBM | **Threshold:** 0.35 | **Features:** 29  
**Test suite:** 55 tests, 55 passing  

---

## Repository Structure

```
part4/
├── app/
│   └── main.py               ← FastAPI application (3 endpoints)
├── tests/
│   ├── __init__.py
│   └── test_api.py           ← 55 pytest test cases
├── model.pkl                 ← Saved model bundle from Part 3
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
└── monitoring_plan.md        ← Post-deployment monitoring & responsible use
```

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/rashiku/part4-d2c-FastAPI.git
cd part4-d2c-FastAPI
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.  
Auto-generated interactive docs: `http://127.0.0.1:8000/docs`

---

## Endpoints

### `GET /health`
Liveness check. Confirms the server is running and the model is loaded.

**Sample response:**
```json
{
  "status": "ok",
  "model": "LightGBM Churn Classifier v1.0",
  "version": "1.0.0",
  "threshold": 0.35,
  "feature_count": 29
}
```

---

### `POST /predict`
Score a single customer for churn risk.

**Input:** 25 raw customer features (the 4 engineered features are computed automatically).  
**Output:** Churn probability, predicted class, risk level, and a plain-language risk explanation.

**Risk levels:**

| Level | Probability Range | Recommended Action |
|---|---|---|
| `low` | < 0.35 | No intervention needed |
| `medium` | 0.35 – 0.60 | Low-cost email nudge |
| `high` | 0.60 – 0.80 | Personalised retention offer |
| `critical` | > 0.80 | Immediate CRM outreach |

**Sample request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST00042",
    "city_tier": "Tier 2",
    "age_group": "25-34",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Silver",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 95,
    "frequency_180d": 1,
    "monetary_180d": 620.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.20,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 1,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 1.0,
    "avg_resolution_hours_90d": 24.0,
    "days_since_signup": 210,
    "sessions_30d": 2,
    "product_views_30d": 5,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 1,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 18
  }'
```

**Sample response:**
```json
{
  "customer_id": "CUST00042",
  "churn_probability": 0.6841,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "High churn risk: high recency (95 days since last order); low recent web activity (last visit 18 days ago); recent support ticket with fully negative sentiment.",
  "threshold_used": 0.35
}
```

---

### `POST /batch_predict`
Score up to 500 customers in a single call.

**Sample request:**
```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { "customer_id": "CUST00001", "city_tier": "Tier 1", ... },
      { "customer_id": "CUST00002", "city_tier": "Tier 3", ... }
    ]
  }'
```

**Sample response:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST00001",
      "churn_probability": 0.1823,
      "predicted_class": 0,
      "risk_level": "low",
      "risk_explanation": "Customer shows healthy engagement ...",
      "threshold_used": 0.35
    },
    {
      "customer_id": "CUST00002",
      "churn_probability": 0.8912,
      "predicted_class": 1,
      "risk_level": "critical",
      "risk_explanation": "Critical churn risk: very high recency (187 days since last order); zero web/app sessions in the last 30 days; no purchases in the last 180 days.",
      "threshold_used": 0.35
    }
  ],
  "total_customers": 2,
  "high_risk_count": 1,
  "processing_time_ms": 12.4
}
```

---

## Input Validation

All inputs are validated by Pydantic before reaching the model.

| Rule | Behaviour on violation |
|---|---|
| Unknown `city_tier` / `age_group` / etc. | HTTP 422 with field-level error message |
| Numeric out of range (e.g. `return_rate > 1.0`) | HTTP 422 |
| Missing required field | HTTP 422 |
| `recency_days > days_since_signup` | HTTP 422 (cross-field validator) |
| Empty batch (`customers: []`) | HTTP 422 |
| Batch exceeding 500 customers | HTTP 422 |

**Valid categorical values:**

| Field | Accepted values |
|---|---|
| `city_tier` | `Tier 1`, `Tier 2`, `Tier 3` |
| `age_group` | `18-24`, `25-34`, `35-44`, `45+` |
| `acquisition_channel` | `Google Search`, `Influencer`, `Instagram`, `Marketplace`, `Organic`, `Referral` |
| `loyalty_tier` | `Gold`, `Silver`, `Platinum`, `Not_Enrolled` |
| `preferred_category` | `Baby Care`, `Fragrance`, `Hair Care`, `Makeup`, `Skin Care`, `Wellness` |
| `marketing_consent` | `Yes`, `No` |

> **Note on loyalty_tier:** Use `"Not_Enrolled"` (not `null` / empty string) for customers not in the loyalty programme.

---

## Loading the Model Directly

```python
import joblib
import pandas as pd

bundle        = joblib.load("model.pkl")
model         = bundle["model"]           # LGBMClassifier
cat_encoders  = bundle["cat_encoders"]    # dict of LabelEncoders
cat_features  = bundle["cat_features"]    # list of categorical column names
all_features  = bundle["all_features"]    # ordered list of all 29 features
threshold     = bundle["best_threshold"]  # 0.35
```

The `all_features` list includes the 4 engineered features that must be computed
before calling `model.predict_proba`:

```python
df["engagement_rate"]    = df["cart_adds_30d"]      / (df["product_views_30d"] + 1)
df["email_click_rate"]   = df["campaign_clicks_30d"] / (df["email_opens_30d"]  + 1)
df["spend_per_order"]    = df["monetary_180d"]       / (df["frequency_180d"]   + 1)
df["recency_x_frequency"]= df["recency_days"].apply(
                               lambda x: __import__("math").log1p(x)
                           ) * (df["frequency_180d"] + 1)
```

---

## Running the Tests

```bash
# From the part4/ root
pytest tests/test_api.py -v
```

Expected output: **55 passed**.

Test coverage:

| Group | Tests | Coverage |
|---|---|---|
| Health endpoint | 5 | Status, schema, threshold, feature count |
| `/predict` — high risk | 8 | Class, probability, risk level, schema, explanation |
| `/predict` — low risk | 5 | Class, probability, risk level, positive language |
| `/batch_predict` | 8 | Count, schema, high_risk_count, order, edge cases |
| Input validation | 11 | All invalid categoricals, out-of-range numerics, missing fields, cross-field |
| Business logic | 8 | Probability bounds, class↔threshold consistency, determinism, batch=individual |
| Edge cases | 6 | New customer, extreme values, all categorical enumerations |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to the saved model bundle |

To use a model in a different location:
```bash
MODEL_PATH=/path/to/your/model.pkl uvicorn app.main:app --reload
```

---

## Dependencies

Key packages (`requirements.txt`):

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
lightgbm==4.3.0
scikit-learn==1.8.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
httpx==0.27.0
pytest==8.2.2
```

> **scikit-learn must be 1.8.0** — this matches the version used when `model.pkl`
> was saved. A lower version will produce an `InconsistentVersionWarning`.

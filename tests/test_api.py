"""
tests/test_api.py
=================
Pytest test suite for the D2C Churn Scoring API.

Run from the repo root:
    pytest tests/test_api.py -v

Requirements: httpx, pytest, pytest-asyncio (all in requirements.txt)
The test client boots the FastAPI app in-process — no running server needed.
"""

import math
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Import the app — TestClient handles lifespan (model loading) automatically
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app, _bundle

# ---------------------------------------------------------------------------
# Session fixture — load model.pkl once before any test runs.
# TestClient at module level does not trigger the FastAPI lifespan, so we
# populate _bundle directly. This mirrors exactly what lifespan does.
# ---------------------------------------------------------------------------
MODEL_PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.pkl")

@pytest.fixture(scope="session", autouse=True)
def preload_model():
    """Load model.pkl into the app's _bundle dict before the test session starts."""
    import joblib
    bundle = joblib.load(MODEL_PKL)
    _bundle.update(bundle)
    yield
    _bundle.clear()

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixture payloads
# ---------------------------------------------------------------------------

# A dormant customer — very high churn risk
HIGH_RISK_PAYLOAD = {
    "customer_id": "CUST_TEST_HIGH",
    "city_tier": "Tier 3",
    "age_group": "45+",
    "acquisition_channel": "Organic",
    "loyalty_tier": "Not_Enrolled",
    "preferred_category": "Skin Care",
    "marketing_consent": "No",
    "recency_days": 180,
    "frequency_180d": 0,
    "monetary_180d": 0.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.1,
    "avg_rating_180d": 0.0,
    "category_diversity_180d": 0,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 400,
    "sessions_30d": 0,
    "product_views_30d": 0,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 45,
}

# A champion customer — very low churn risk
LOW_RISK_PAYLOAD = {
    "customer_id": "CUST_TEST_LOW",
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Platinum",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 4,
    "frequency_180d": 6,
    "monetary_180d": 4200.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.08,
    "avg_rating_180d": 4.9,
    "category_diversity_180d": 3,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 450,
    "sessions_30d": 18,
    "product_views_30d": 52,
    "cart_adds_30d": 5,
    "wishlist_adds_30d": 3,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 6,
    "campaign_clicks_30d": 3,
    "last_visit_days_ago": 1,
}

# A borderline / medium-risk customer
MEDIUM_RISK_PAYLOAD = {
    "customer_id": "CUST_TEST_MED",
    "city_tier": "Tier 2",
    "age_group": "35-44",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Silver",
    "preferred_category": "Hair Care",
    "marketing_consent": "Yes",
    "recency_days": 55,
    "frequency_180d": 2,
    "monetary_180d": 780.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.25,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 2,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 12.0,
    "days_since_signup": 280,
    "sessions_30d": 3,
    "product_views_30d": 8,
    "cart_adds_30d": 1,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 10,
}


# ===========================================================================
# TEST 1 — Health endpoint
# ===========================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self):
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self):
        """Response must contain status='ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_contains_model_info(self):
        """Response must include model name, version, threshold, feature_count."""
        data = client.get("/health").json()
        assert "model" in data
        assert "version" in data
        assert "threshold" in data
        assert "feature_count" in data

    def test_health_threshold_value(self):
        """Threshold must be 0.35 (matches model.pkl best_threshold)."""
        data = client.get("/health").json()
        assert data["threshold"] == pytest.approx(0.35, abs=0.01)

    def test_health_feature_count(self):
        """Feature count must be 29 (25 base + 4 engineered)."""
        data = client.get("/health").json()
        assert data["feature_count"] == 29


# ===========================================================================
# TEST 2 — /predict: high-risk customer
# ===========================================================================

class TestPredictHighRisk:

    def test_predict_returns_200(self):
        """POST /predict with valid payload must return HTTP 200."""
        response = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200

    def test_predict_high_risk_class(self):
        """Dormant customer with 0 sessions and 180-day recency must be predicted as churn."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert data["predicted_class"] == 1

    def test_predict_high_risk_probability(self):
        """Churn probability for dormant customer must exceed 0.50."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert data["churn_probability"] > 0.50

    def test_predict_high_risk_level(self):
        """Risk level must be 'high' or 'critical' for a dormant customer."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert data["risk_level"] in ("high", "critical")

    def test_predict_response_schema(self):
        """Response must contain all required fields."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        required_fields = {
            "customer_id", "churn_probability", "predicted_class",
            "risk_level", "risk_explanation", "threshold_used"
        }
        assert required_fields.issubset(data.keys())

    def test_predict_customer_id_echoed(self):
        """customer_id from the request must be echoed back in the response."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert data["customer_id"] == HIGH_RISK_PAYLOAD["customer_id"]

    def test_predict_explanation_nonempty(self):
        """Risk explanation must be a non-empty string."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert isinstance(data["risk_explanation"], str)
        assert len(data["risk_explanation"]) > 10

    def test_predict_threshold_in_response(self):
        """threshold_used in response must equal 0.35."""
        data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert data["threshold_used"] == pytest.approx(0.35, abs=0.01)


# ===========================================================================
# TEST 3 — /predict: low-risk customer
# ===========================================================================

class TestPredictLowRisk:

    def test_predict_low_risk_class(self):
        """Active champion customer must be predicted as retained (class=0)."""
        data = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        assert data["predicted_class"] == 0

    def test_predict_low_risk_probability(self):
        """Churn probability for champion must be below 0.35."""
        data = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        assert data["churn_probability"] < 0.35

    def test_predict_low_risk_level(self):
        """Risk level for champion customer must be 'low'."""
        data = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        assert data["risk_level"] == "low"

    def test_predict_low_risk_explanation_positive(self):
        """Explanation for low-risk customer must contain positive language."""
        data = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        explanation = data["risk_explanation"].lower()
        # Should contain positive framing, not churn alarm language
        assert any(word in explanation for word in
                   ["healthy", "low churn", "active", "engagement"])

    def test_predict_without_customer_id(self):
        """Request without customer_id should still succeed (field is optional)."""
        payload = {k: v for k, v in LOW_RISK_PAYLOAD.items() if k != "customer_id"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] is None


# ===========================================================================
# TEST 4 — /batch_predict
# ===========================================================================

class TestBatchPredict:

    def test_batch_returns_200(self):
        """POST /batch_predict with valid payloads must return HTTP 200."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, MEDIUM_RISK_PAYLOAD]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self):
        """total_customers in response must match input count."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, MEDIUM_RISK_PAYLOAD]}
        data = client.post("/batch_predict", json=payload).json()
        assert data["total_customers"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_predictions_schema(self):
        """Each prediction in the batch must have all required fields."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        data = client.post("/batch_predict", json=payload).json()
        required = {"customer_id", "churn_probability", "predicted_class",
                    "risk_level", "risk_explanation", "threshold_used"}
        for pred in data["predictions"]:
            assert required.issubset(pred.keys())

    def test_batch_high_risk_count(self):
        """high_risk_count must correctly count high + critical risk customers."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, HIGH_RISK_PAYLOAD]}
        data = client.post("/batch_predict", json=payload).json()
        # Both HIGH_RISK customers should be flagged; LOW_RISK should not
        assert data["high_risk_count"] >= 2

    def test_batch_processing_time_present(self):
        """processing_time_ms must be a non-negative float."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        data = client.post("/batch_predict", json=payload).json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0.0

    def test_batch_order_preserved(self):
        """Predictions must be returned in the same order as the input customers."""
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        data = client.post("/batch_predict", json=payload).json()
        preds = data["predictions"]
        assert preds[0]["customer_id"] == HIGH_RISK_PAYLOAD["customer_id"]
        assert preds[1]["customer_id"] == LOW_RISK_PAYLOAD["customer_id"]

    def test_batch_single_customer(self):
        """Batch with exactly one customer must work (min_length=1)."""
        payload = {"customers": [LOW_RISK_PAYLOAD]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        assert response.json()["total_customers"] == 1

    def test_batch_empty_list_rejected(self):
        """Batch with zero customers must be rejected with 422."""
        response = client.post("/batch_predict", json={"customers": []})
        assert response.status_code == 422


# ===========================================================================
# TEST 5 — Input validation (invalid payloads)
# ===========================================================================

class TestInputValidation:

    def test_invalid_city_tier_rejected(self):
        """Unknown city_tier value must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "city_tier": "Tier 9"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_group_rejected(self):
        """Unknown age_group must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "age_group": "100+"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_acquisition_channel_rejected(self):
        """Unknown acquisition_channel must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "acquisition_channel": "TikTok"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_loyalty_tier_rejected(self):
        """Unknown loyalty_tier must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "loyalty_tier": "Diamond"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_preferred_category_rejected(self):
        """Unknown preferred_category must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "preferred_category": "Pet Care"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_marketing_consent_rejected(self):
        """marketing_consent must be 'Yes' or 'No' — anything else is rejected."""
        payload = {**HIGH_RISK_PAYLOAD, "marketing_consent": "Maybe"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_recency_rejected(self):
        """Negative recency_days must return HTTP 422 (ge=0 constraint)."""
        payload = {**HIGH_RISK_PAYLOAD, "recency_days": -5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_return_rate_above_1_rejected(self):
        """return_rate_180d > 1.0 must return HTTP 422 (le=1.0 constraint)."""
        payload = {**HIGH_RISK_PAYLOAD, "return_rate_180d": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_discount_above_1_rejected(self):
        """avg_discount_pct_180d > 1.0 must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "avg_discount_pct_180d": 2.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rating_above_5_rejected(self):
        """avg_rating_180d > 5.0 must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "avg_rating_180d": 6.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_field_rejected(self):
        """Omitting a required field (recency_days) must return HTTP 422."""
        payload = {k: v for k, v in HIGH_RISK_PAYLOAD.items() if k != "recency_days"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_multiple_fields_rejected(self):
        """Omitting multiple required fields must return HTTP 422."""
        payload = {"customer_id": "CUST_INCOMPLETE", "city_tier": "Tier 1"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_cross_field_recency_exceeds_signup(self):
        """recency_days > days_since_signup must return HTTP 422 (cross-field validator)."""
        payload = {**HIGH_RISK_PAYLOAD, "recency_days": 500, "days_since_signup": 100}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_string_for_numeric(self):
        """Passing a string where a float is expected must return HTTP 422."""
        payload = {**HIGH_RISK_PAYLOAD, "recency_days": "ninety"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_completely_empty_body_rejected(self):
        """Empty JSON body must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ===========================================================================
# TEST 6 — Business logic & probability consistency
# ===========================================================================

class TestBusinessLogic:

    def test_probability_between_0_and_1(self):
        """Churn probability must always be in [0, 1]."""
        for payload in [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, MEDIUM_RISK_PAYLOAD]:
            data = client.post("/predict", json=payload).json()
            assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predicted_class_matches_threshold(self):
        """predicted_class must be 1 iff churn_probability >= threshold_used."""
        for payload in [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, MEDIUM_RISK_PAYLOAD]:
            data = client.post("/predict", json=payload).json()
            expected_class = 1 if data["churn_probability"] >= data["threshold_used"] else 0
            assert data["predicted_class"] == expected_class, (
                f"Class mismatch: proba={data['churn_probability']}, "
                f"threshold={data['threshold_used']}, class={data['predicted_class']}"
            )

    def test_risk_level_matches_probability(self):
        """risk_level must be consistent with churn_probability buckets."""
        for payload in [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, MEDIUM_RISK_PAYLOAD]:
            data = client.post("/predict", json=payload).json()
            p = data["churn_probability"]
            level = data["risk_level"]
            if p < 0.35:
                assert level == "low",      f"Expected low for p={p}, got {level}"
            elif p < 0.60:
                assert level == "medium",   f"Expected medium for p={p}, got {level}"
            elif p < 0.80:
                assert level == "high",     f"Expected high for p={p}, got {level}"
            else:
                assert level == "critical", f"Expected critical for p={p}, got {level}"

    def test_high_risk_higher_probability_than_low_risk(self):
        """Dormant customer must have strictly higher churn probability than champion."""
        high_data = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        low_data  = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        assert high_data["churn_probability"] > low_data["churn_probability"]

    def test_deterministic_predictions(self):
        """Identical inputs must produce identical predictions (model is deterministic)."""
        r1 = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        r2 = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert r1["churn_probability"] == r2["churn_probability"]
        assert r1["predicted_class"]   == r2["predicted_class"]

    def test_batch_matches_individual_predictions(self):
        """Batch predictions must match individual /predict calls for same inputs."""
        batch_resp = client.post("/batch_predict", json={
            "customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]
        }).json()

        single_high = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        single_low  = client.post("/predict", json=LOW_RISK_PAYLOAD).json()

        batch_preds = batch_resp["predictions"]
        assert batch_preds[0]["churn_probability"] == pytest.approx(
            single_high["churn_probability"], abs=1e-4)
        assert batch_preds[1]["churn_probability"] == pytest.approx(
            single_low["churn_probability"], abs=1e-4)

    def test_not_enrolled_loyalty_accepted(self):
        """'Not_Enrolled' is a valid loyalty_tier value and must not be rejected."""
        payload = {**LOW_RISK_PAYLOAD, "loyalty_tier": "Not_Enrolled"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_zero_monetary_zero_frequency_accepted(self):
        """Customers with no 180d purchases (monetary=0, frequency=0) must score without error."""
        payload = {**HIGH_RISK_PAYLOAD, "monetary_180d": 0.0, "frequency_180d": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ===========================================================================
# TEST 7 — Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_brand_new_customer(self):
        """Customer with days_since_signup=1 and no history must score without error."""
        payload = {
            **HIGH_RISK_PAYLOAD,
            "days_since_signup": 1,
            "recency_days": 1,
            "frequency_180d": 1,
            "monetary_180d": 500.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_maximum_sessions(self):
        """Very high session count must not cause overflow or error."""
        payload = {**LOW_RISK_PAYLOAD, "sessions_30d": 999, "product_views_30d": 9999}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_gold_loyalty_tier_accepted(self):
        """All valid loyalty tiers must be accepted."""
        for tier in ["Gold", "Silver", "Platinum", "Not_Enrolled"]:
            payload = {**LOW_RISK_PAYLOAD, "loyalty_tier": tier}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for loyalty_tier={tier}"

    def test_all_acquisition_channels_accepted(self):
        """All 6 valid acquisition channels must be accepted."""
        channels = ["Google Search", "Influencer", "Instagram",
                    "Marketplace", "Organic", "Referral"]
        for ch in channels:
            payload = {**LOW_RISK_PAYLOAD, "acquisition_channel": ch}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for channel={ch}"

    def test_all_age_groups_accepted(self):
        """All 4 valid age groups must be accepted."""
        for ag in ["18-24", "25-34", "35-44", "45+"]:
            payload = {**LOW_RISK_PAYLOAD, "age_group": ag}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for age_group={ag}"

    def test_high_return_rate_unhappy_customer(self):
        """Customer with high return rate and negative tickets must score without error."""
        payload = {
            **MEDIUM_RISK_PAYLOAD,
            "return_rate_180d": 1.0,
            "ticket_count_90d": 3,
            "negative_ticket_rate_90d": 1.0,
            "avg_resolution_hours_90d": 72.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # High returns + negative tickets should push toward churn
        assert data["churn_probability"] > 0.3

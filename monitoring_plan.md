# Monitoring Plan & Responsible Use
## D2C Churn Scoring API — Post-Deployment Guide

**Model:** LightGBM Churn Classifier v1.0  
**Snapshot date:** 2025-09-30 | **Threshold:** 0.35  
**Prediction window:** 60 days post-scoring

---

## 1. What to Monitor After Deployment

Monitoring falls into four layers: **data**, **predictions**, **business outcomes**,
and **API health**. Each requires a different cadence and alert threshold.

---

### 1.1 Data Drift — Are Inputs Changing?

The model was trained on customer behaviour as of 2025-09-30. If the distribution
of input features shifts significantly, predictions will degrade even if the model
code is unchanged.

| Feature | Why It Drifts | Monitor With |
|---|---|---|
| `recency_days` | Seasonality, sale events compress recency | KS-statistic vs training distribution |
| `sessions_30d` | App redesign, marketing campaigns spike sessions | Mean ± 2 std dev check monthly |
| `monetary_180d` | Price changes, new product launches, inflation | Percentile shift (P50, P90) monthly |
| `loyalty_tier` distribution | Loyalty programme expansion or restructuring | Chi-squared test on tier proportions |
| `acquisition_channel` mix | New paid channels added, organic traffic changes | Chi-squared test monthly |

**Alert threshold:** Raise a flag if the KS-statistic for any top-5 SHAP feature
exceeds **0.10** compared to the training distribution, on two consecutive monthly
checks.

**Action:** When drift is detected, run a shadow evaluation: score the current
cohort with both the existing model and a freshly retrained candidate. If the
candidate's precision/recall gap is > 5pp, promote the retrained model.

---

### 1.2 Prediction Distribution — Is the Model Behaving Normally?

Even without labelled outcomes, you can monitor the model's output distribution
to catch silent failures.

| Signal | Expected Range | Alert If |
|---|---|---|
| Mean predicted churn probability | 0.42–0.52 (mirrors training base rate) | Deviates by > 5pp for 2 consecutive months |
| % customers scored as `critical` (proba > 0.80) | 10–20% | Exceeds 30% or drops below 5% |
| % customers scored as `low` (proba < 0.35) | 35–50% | Falls below 20% |
| Batch processing time (`processing_time_ms`) | < 500ms for 500 customers | Exceeds 2,000ms |
| Prediction variance across cohort | Check monthly | Near-zero variance → model returning constant scores |

**Why this matters:** A spike in `critical` scores after a sale event is expected
(recency compressed). A persistent spike with no obvious business event signals
either data pipeline corruption or concept drift.

---

### 1.3 Business Outcomes — Is the Model Correct? (60-day lag)

This is the ground truth check. For every cohort scored in month M, wait 60 days
and compare predictions against actual purchase behaviour.

| Metric | Target | Alert Threshold | Cadence |
|---|---|---|---|
| Recall on labelled cohort | ≥ 0.85 | < 0.80 for 2 months | Monthly (60d lag) |
| Precision on labelled cohort | ≥ 0.65 | < 0.55 for 2 months | Monthly (60d lag) |
| False negative rate | ≤ 8% | > 12% | Monthly (60d lag) |
| Campaign conversion rate (churners retained) | ≥ 20% | < 12% | Monthly |
| Revenue recovered per ₹ of campaign spend | ≥ 5× | < 2× | Quarterly |

**Holdout group:** Always keep a random 10% of high-risk customers uncontacted
(control group). This is the only way to measure the model's true causal impact
rather than just correlation.

---

### 1.4 API Health — Is the Service Running Reliably?

| Signal | Alert Threshold | Action |
|---|---|---|
| HTTP 5xx error rate | > 1% of requests in any 15-min window | Page on-call engineer |
| HTTP 422 validation error rate | > 15% of requests | Investigate upstream data pipeline |
| `/health` endpoint response time | > 200ms | Alert; potential model reload issue |
| Model load failure at startup | Any occurrence | Block deployment; page engineer |
| Request volume drop | > 50% vs 7-day average | Check CRM integration |

---

## 2. Retraining Triggers

Retrain the model when **any one** of the following conditions is met:

| Trigger | Description |
|---|---|
| **Scheduled** | Every 90 days regardless of performance — customer behaviour evolves |
| **Recall drops** | Recall on labelled cohort falls below 0.80 on two consecutive monthly evaluations |
| **Precision drops** | Precision falls below 0.55 on two consecutive monthly evaluations |
| **Feature drift** | KS-statistic > 0.10 on any top-5 SHAP feature for two consecutive months |
| **Business event** | Major product launch, pricing restructure, new acquisition channel, or loyalty programme overhaul |
| **Data pipeline change** | Any change to how features are computed (lookback windows, aggregation logic) |

**Retraining process:**
1. Compute new snapshot features using the same `rfm_modeling_snapshot.csv` logic
   on the most recent 6 months of orders.
2. Retrain on the full available history (not just the new 6 months).
3. Evaluate on a held-out recent cohort — do not reuse the original test split.
4. If new model beats the current model on both recall (≥ 0.85) and precision
   (≥ 0.65), promote it. Otherwise keep the current model and investigate.
5. Document the retraining run: date, training size, evaluation metrics, and
   reason for retraining in a model version log.

---

## 3. Responsible Use Policy

This section defines how the API output **should** and **should not** be used
by the retention team and any downstream systems.

---

### 3.1 Permitted Uses

- **Prioritising outreach queues:** Sort the customer list by `churn_probability`
  descending. Work top-down within campaign budget constraints.
- **Routing intervention type:** Use `risk_level` to match intensity of outreach
  to risk — email for `medium`, personal call for `critical`.
- **Campaign budget allocation:** Use `high_risk_count` from `/batch_predict`
  to size the campaign budget needed before the CRM manager approves spend.
- **A/B testing:** Score all customers, randomise a holdout group, and measure
  true retention lift from interventions.
- **Weekly reporting:** Track the distribution of `risk_level` over time to
  surface early signs of cohort-level churn acceleration.

---

### 3.2 Prohibited Uses

| Action | Why It Is Prohibited |
|---|---|
| **Automatically refusing customers service** based on a high churn score | A churn score is a prediction about future behaviour, not evidence of current intent. Customers must not be penalised for a model output. |
| **Downgrading loyalty tier** of high-risk customers | Reducing loyalty benefits for at-risk customers is precisely the wrong intervention — it accelerates churn. |
| **Sharing the score with customers** | Telling a customer "our model thinks you will leave" is damaging to the relationship and likely to accelerate the outcome being predicted. |
| **Sending intrusive interventions to `marketing_consent = No` customers** | Customers who have not opted in to marketing must not receive promotional campaigns regardless of their churn score. |
| **Using score as the sole input for high-value decisions** | For customers with lifetime spend > ₹10,000 or `risk_level = critical`, always combine the model score with a human review before sending an expensive offer. |
| **Deploying predictions older than 30 days** | The model uses a 30-day web activity window. A score computed 31+ days ago may no longer reflect the customer's current state. Re-score before acting. |
| **Applying the model to B2B or wholesale accounts** | The model was trained on D2C consumer behaviour. Business account purchase patterns are structurally different. |

---

### 3.3 Fairness & Demographic Considerations

The model includes `age_group` and `city_tier` as features. This means predictions
are partially conditioned on demographic attributes.

**Risk:** If retention campaign budgets are capped, allocating them purely by model
score could systematically under-serve demographic groups with higher base churn
rates — through no fault of individual behaviour.

**Required safeguard:** Before each campaign launch, the CRM manager must verify
that outreach coverage does not fall below **15% of any single `age_group` or
`city_tier`** segment. If the budget does not stretch to that floor, reduce the
intervention intensity (e.g. email instead of call) before cutting coverage.

---

### 3.4 Borderline Zone Policy

Customers with `churn_probability` between **0.35–0.50** are in the model's
uncertainty zone — near the decision boundary where small feature changes flip
the prediction. These customers should:

1. Receive **lighter-touch interventions** (email only, no large discounts) rather
   than full retention campaigns.
2. Be **re-scored after 14 days** if they have not responded, rather than assuming
   the first prediction was correct.
3. Never be sent aggressive discount offers purely on the basis of a borderline score.

---

### 3.5 Transparency to Customers

If a customer asks why they received a retention offer or asks whether they are
being tracked:

- **Do:** Acknowledge that the brand uses purchase history and engagement data
  to personalise outreach.
- **Do:** Refer to the brand's privacy policy and data usage terms.
- **Do not:** Share the specific churn probability score or the model's reasoning.
- **Do not:** Deny that personalisation is happening.

---

## 4. Incident Response

If the API returns unexpected results (e.g. all customers scored as `critical`,
all scores = 0.0, or systematic 5xx errors):

1. **Stop campaign sends immediately** — do not act on potentially corrupted scores.
2. Call `GET /health` — if it returns anything other than `{"status": "ok"}`,
   the model has not loaded correctly. Restart the service.
3. Check the data pipeline: confirm that input features are being computed with
   the correct lookback windows and that `loyalty_tier` nulls are being filled
   with `"Not_Enrolled"` before calling the API.
4. If the issue persists, fall back to the manual CRM priority bucket
   (`manual_priority_bucket` from Part 2 `segments.csv`) as a temporary targeting
   signal while the model is investigated.
5. Document the incident: timestamp, symptoms, affected cohort size, root cause,
   and resolution in the model version log.

---

*Monitoring plan prepared for the D2C Churn Scoring API v1.0.*  
*Review and update this document at each retraining cycle.*

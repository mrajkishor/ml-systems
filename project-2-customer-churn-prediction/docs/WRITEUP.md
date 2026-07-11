# Write-up: Decisions and Tradeoffs

## Data

The Telco Customer Churn dataset is normally downloaded from Kaggle with an account and API
token. `get_dataset()` follows the same pattern as Project 1: prefer a manually downloaded
`data/raw/train.csv` if present, otherwise fetch the identical dataset via scikit-learn's
OpenML mirror (`fetch_openml(data_id=42178)`) -- 7043 rows, 20 columns, same data. Zero
credentials needed for a clean clone to run end to end.

## EDA findings that shaped everything downstream

- **26.54% churn rate** (95% Wilson CI: 25.5%-27.6%). Imbalanced, but not extreme -- this
  matters because it means both `class_weight='balanced'` and SMOTE are viable strategies
  (an 0.1%-positive fraud dataset would rule out plain class weighting much faster).
- **Tenure is the strongest signal by a wide margin**: churned customers average 18.0 months
  tenure vs 37.6 months for retained customers (Welch's t=34.8, p≈1.2e-232). New customers
  churn; customers who've stuck around for years mostly keep sticking around.
- **Contract type is hugely predictive** (chi2=1184.6, p≈6e-258 against `Churn`). This is
  exactly why `Contract` gets ordinal-encoded (`Month-to-month` < `One year` < `Two year`)
  rather than one-hot: the order itself carries information a tree model can split on
  directly, and a linear model can weight as a single monotonic signal instead of three
  independent dummy coefficients that happen to correlate.

## Feature engineering decisions

- **The `TotalCharges` blank-string quirk.** 11 rows -- all with `tenure=0` -- store
  `TotalCharges` as the literal string `' '` instead of a number, because those customers
  haven't been billed yet. `pd.to_numeric(..., errors="coerce")` turns those into `NaN`,
  which then get filled with `0` (not the column median) since the correct value for a
  brand-new customer's total charges genuinely is zero, not "unknown."
- **Binary encoding for Yes/No fields, ordinal for `Contract`, one-hot for everything else.**
  Fields like `Partner`, `Dependents`, `PhoneService` are strictly binary, so mapping them to
  0/1 avoids doubling the one-hot column count for no benefit. `Contract` gets the ordinal
  treatment described above. Everything else (`InternetService`, `PaymentMethod`, the
  `OnlineSecurity`/`TechSupport`/etc. family with their three-way `Yes`/`No`/`No internet
  service` categories) is genuinely nominal and gets one-hot encoded.
- **Stratified train/test split**, not a plain random split. With a 26.5% minority class, an
  unlucky random split can meaningfully shift the class balance between train and test;
  `train_test_split(..., stratify=y)` keeps both splits at ~26.5% churn.

## Imbalance handling: class weights vs SMOTE

Both were benchmarked on the same tuned LightGBM architecture:

| Strategy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| `class_weight='balanced'` | 0.607 | 0.666 | 0.635 | 0.841 |
| SMOTE oversampling | 0.533 | 0.759 | 0.626 | 0.839 |

They land at almost the same ROC-AUC but at very different points on the precision/recall
tradeoff. Class weighting penalizes misclassifying the minority class during training without
inventing new data points; SMOTE synthesizes new minority-class examples by interpolating
between real ones, which tends to push recall up at precision's expense. **Neither is
objectively better** -- the right choice depends on whether missing a churner or bothering a
non-churner costs more, which is exactly the business question the cost-sensitive analysis
below answers. One implementation detail worth flagging: SMOTE is applied only to the
training split (`apply_smote` is never called on validation/test data) -- resampling before
a split would let synthetic points derived from test-set neighbors leak into training.

## Modeling results

LightGBM (tuned) edges out Logistic Regression on ROC-AUC (0.841 vs 0.837) and clearly wins
on F1 (0.635 vs 0.624), but Logistic Regression's *untuned* CV ROC-AUC (0.840) was competitive
with tuned LightGBM before any tuning happened at all -- a reminder that on a small (7043-row),
mostly-categorical dataset with strong linear signal (tenure, contract type), a simple
regularized linear model is a genuinely strong baseline, not just a placeholder to beat.
CatBoost's untuned recall (0.813) and PR-AUC (0.647) were actually the best of any single
model before tuning -- if recall were the deployment priority (catch as many churners as
possible, precision be damned), CatBoost's defaults would be the better starting point than
LightGBM.

**Tuning (RandomizedSearchCV, 30 iterations, stratified 5-fold, ROC-AUC scoring)** moved
LightGBM's test ROC-AUC from 0.826 to 0.841 and F1 from 0.613 to 0.635. The winning
hyperparameters (`max_depth=2`, `n_estimators=443`, `learning_rate=0.032`) favor many shallow,
slow-learning trees over fewer deep ones -- the same bias-toward-simplicity pattern that won
in Project 1's XGBoost tuning, and a reasonable prior for small tabular datasets in general.

**Learning curve:** training and validation ROC-AUC converge rather than diverging as training
size grows (see `docs/images/learning_curve.png`), meaning this model is closer to
underfitting than overfitting at this data size -- more data would likely help more than more
regularization would.

## Business interpretation: threshold choice is a cost decision, not a metric decision

The headline result of this project: **the F1-optimal threshold (0.64) costs more than the
naive default (0.50)**, and both cost more than the threshold chosen to directly minimize
expected dollar cost (0.23). F1 treats a false positive and false negative as equally bad by
construction; in this business, they aren't -- missing a churner costs an assumed $800 in lost
lifetime value, while a wasted retention offer to a customer who wouldn't have churned costs
an assumed $50. Optimizing for F1 implicitly assumes a 1:1 cost ratio that doesn't hold here,
so it picks a threshold that's too conservative about flagging churn risk. The cost-optimal
threshold deliberately accepts far more false positives (533 vs 161) to cut false negatives
from 125 to 18, because at these assumed costs, that trade is worth it.

**Caveat, stated plainly:** `retention_cost=$50`, `customer_value_lost=$800`, and
`retention_success_rate=50%` in `business.py` are illustrative placeholders, not fitted from
real data. Before using this for an actual retention-campaign budget decision, they need to be
replaced with real numbers: actual average customer lifetime value (segmented by contract
type and tenure, since those clearly aren't uniform), the actual cost of whatever retention
incentive is being offered, and a measured save rate from a past campaign rather than an
assumed 50/50 coin flip.

## Serving design

The Streamlit dashboard exposes all 19 raw input fields directly (not a reduced subset) since
there are few enough to fit a two-column form comfortably, and it means the exact same
`build_feature_pipeline` used in training runs unmodified at inference time -- no separate
"lite" preprocessing path to keep in sync. Predictions are made at the persisted cost-optimal
threshold (`artifacts/threshold.joblib`), not the sklearn default of 0.5, so the dashboard's
Churn/No-churn call reflects the business framing above rather than an arbitrary cutoff.
Explanations use SHAP's `TreeExplainer` on the same tuned LightGBM model, shown as a per-feature
contribution bar chart for that one prediction -- not a global importance ranking, since the
question a user of this dashboard actually has is "why did *this* customer get flagged," not
"what matters on average."

## What's next

- Deploy the dashboard to Streamlit Community Cloud or Render for a live demo link (needs an
  account -- see main README caveat).
- Replace the illustrative cost assumptions with real customer lifetime value and retention
  campaign data if this were driving an actual business decision.
- Try a stacked ensemble of Logistic Regression + LightGBM given how close their standalone
  scores land, similar to the follow-up noted in Project 1.

# Write-up: Decisions and Tradeoffs

## Data

Kaggle's House Prices competition requires an account + API token to download directly.
Rather than gate the project on that, `get_dataset()` prefers a manually downloaded
`data/raw/train.csv` if you have one, but falls back to fetching the identical dataset via
scikit-learn's OpenML mirror (`fetch_openml(name="house_prices")`) — same 1460 rows, 81
columns, `SalePrice` included. This makes the whole pipeline reproducible from a clean clone
with zero credentials.

## EDA findings that shaped everything downstream

- `SalePrice` is right-skewed (a small number of very expensive homes). This is why every
  model is trained on `log1p(SalePrice)` and predictions are inverted with `expm1` before
  computing dollar-denominated metrics — training on the raw price would let the few
  expensive-home errors dominate the loss.
- `OverallQual` correlates with `SalePrice` at r=0.79 (p≈0, essentially certain not to be
  noise). `CentralAir` (Y/N) splits the data into two clearly different price populations:
  $186k mean vs $105k mean, Welch's t=17.3, p≈2e-37. Both findings reappear later —
  `OverallQual` is the single most important SHAP feature, and quality/amenity fields
  dominate the top-10 list.
- Two homes have `GrLivArea` > 4000 but sold for under $300k (documented in the original
  De Cock paper as data-entry anomalies, not genuine market outcomes). Every other
  large home sold in the $700k+ range consistent with its size. These two rows are
  dropped before training — keeping them would teach the model that enormous homes are
  sometimes nearly free, which isn't a pattern worth learning.

## Feature engineering decisions

- **Missingness is mostly meaningful, not random.** In the Ames data dictionary, `NaN` in
  columns like `PoolQC`, `FireplaceQu`, `GarageType` means "this house doesn't have that
  feature," not "value unknown." Filling these with the column mode or a generic imputer
  would actively distort the signal, so `features/missing.py` fills them with an explicit
  `"None"` category (or `0` for the matching numeric columns) before any statistical
  imputation runs. Only genuinely missing values (`LotFrontage`, `Electrical`) fall through
  to median/mode imputation — `LotFrontage` specifically by neighborhood median, since
  frontage varies a lot by area but is fairly consistent within one.
- **Ordinal encoding, not one-hot, for quality scales.** Columns like `KitchenQual` or
  `ExterQual` have a real order (`Po` < `Fa` < `TA` < `Gd` < `Ex`). One-hot encoding throws
  that ordering away and forces the model to re-learn it from scratch across separate
  columns; mapping to integers 0–5 preserves it directly and shrinks the feature count. All
  other nominal categoricals (`Neighborhood`, `SaleCondition`, etc.) still get one-hot
  encoded — there's no natural order to `Neighborhood` values.
- **Split before fitting the pipeline, not after.** `prepare_train_test_data()` splits raw
  rows into train/test first, then calls `.fit_transform()` on train and `.transform()`
  (not fit) on test. Fitting the scaler/imputer/encoder on the full dataset before splitting
  is a common and easy-to-miss leakage bug — it lets test-set statistics (e.g. the median
  used to impute a training row) leak backward into training.

## Modeling results — and an uncomfortable one

Lasso and Ridge outperformed both Random Forest and tuned XGBoost on the held-out test set
(R²=0.932 vs 0.927). This is a real result, not a bug: after the feature engineering above
(especially ordinal encoding + log-target), the relationship between features and
`log(SalePrice)` is close to linear, and Ames Housing (1460 rows, curated by a statistics
professor for a textbook) is small and clean enough that regularized linear models don't
have much bias to overcome, while tree ensembles have more variance to spend their extra
flexibility on with this little data.

**The API still serves the tuned XGBoost model, not Lasso.** That's a deliberate choice for
this portfolio project: it lets the full pipeline (Optuna tuning + SHAP TreeExplainer) run
end-to-end, and demonstrates handling a more complex model family. If this were a real
production decision instead of a demonstration of technique breadth, **Lasso would win** —
it's simpler, faster to serve, and scored better here. That tradeoff (breadth of technique
vs. actually-best-model) is worth stating plainly rather than picking XGBoost and pretending
it was the empirically superior choice.

Optuna's 30-trial TPE search still meaningfully improved XGBoost (MAE $14,773 → $14,409,
−2.5%) by finding a shallower tree depth (4 vs the default-ish 3) paired with a lower
learning rate and more estimators — a classic tuning outcome for gradient boosting on a
small dataset: more, smaller steps generalize better than fewer, bigger ones.

**Bias/variance read from the 5-fold CV std:** Random Forest had both the highest mean CV
RMSE and one of the highest stds — a sign of underfitting a linear-ish target with default
hyperparameters rather than overfitting. Lasso/Ridge had the lowest std, i.e. the most
stable estimates across folds, mirroring their test-set win.

## Serving design

The `/predict` endpoint accepts a partial JSON dict of raw feature names rather than a fixed
schema of all ~79 training columns — nobody hand-typing a request wants to fill in `PoolQC`
and `MiscFeature`. Missing fields are reindexed to `NaN` and pass through the same imputation
the training pipeline uses, so a request with just `{"OverallQual": 7, "GrLivArea": 1800}`
still gets a reasonable prediction, just built on imputed defaults for everything else.

The fitted pipeline, model, and the exact list of training columns are persisted together
(`artifacts/feature_pipeline.joblib`, `model.joblib`, `feature_columns.joblib`) so inference
never silently uses a different preprocessing path than training did.

## What's next

- Deploy to Railway/Render (needs an actual account — see main README) for a live demo link.
- If this were going to production, swap the served model to Lasso given the test-set
  result above, or at minimum A/B the two.
- The current 5 candidate models don't include a stacked/blended ensemble — worth trying
  given how close Lasso, Ridge, and XGBoost land on this dataset.

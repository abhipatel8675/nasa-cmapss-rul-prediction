# Modeling & Data Decisions

Short rationales for every major choice in this C-MAPSS FD001 RUL project.

| Decision | Choice | Why |
|----------|--------|-----|
| Dataset | NASA C-MAPSS **FD001 only** | Real public benchmark; single operating condition / fault mode keeps the pipeline clear and reproducible for a one-day portfolio delivery. |
| Task type | **Regression** on RUL | Matches the NASA problem definition (remaining cycles), not a binary fail/healthy classifier. |
| Cleaning | Drop constant sensors/settings | On FD001 they have ~zero variance and add noise / collinearity without signal. |
| RUL cap | **125** cycles for training | Standard piecewise-linear target in C-MAPSS literature; prevents early healthy cycles from dominating loss. Test metrics still use uncapped labels. |
| Rolling window | **5** cycles, causal mean/std/slope | Captures short-term degradation dynamics without looking ahead; small window limits burn-in on short test trajectories. |
| Features | Raw kept signals + rolling stats | Gives linear models usable trends and lets trees pick non-linear combinations. |
| Preprocessing | `StandardScaler` for linear model only | Linear models need comparable scales; tree models are scale-invariant. Scaler is fit on **train engines only**. |
| Split | **Engine-wise** train/val + official NASA test | Rows from the same engine are dependent; row-shuffle CV would leak and inflate scores. |
| Validation size | 20% of training engines | Enough engines for a stable holdout while leaving most data for fitting. |
| Baseline | Linear Regression (scaled) | Simple, interpretable reference every stronger model must beat. |
| Strong models | Random Forest + XGBoost | Strong tabular baselines for non-linear sensor interactions without deep-learning complexity. |
| Tuning | `RandomizedSearchCV` + **GroupKFold** on train engines | Searches useful hyperparameters without putting validation engines into the search folds. |
| Test protocol | Predict on **last cycle** per test engine | Matches the official PHM evaluation setup and `RUL_FD001.txt`. |
| Metrics | MAE, RMSE, **NASA asymmetric score** | MAE/RMSE are standard; NASA score encodes higher cost for late (optimistic) RUL predictions. |
| Error analysis | Residuals, error vs true RUL, worst engines | Surfaces *where* the model fails, not only average scores. |
| Interpretation | Impurity / gain feature importance | Explains which sensors and rolling stats drive predictions for stakeholders. |
| Domain risks | Leakage, metric choice, early vs late errors | Healthcare class-imbalance / FN-FP framing does not apply; these are the RUL analogues we explicitly control. |
| Scope exclusion | No LSTM / FD002–004 / serving API | Keeps the repo focused, honest, and deliverable without diluting the core ML story. |

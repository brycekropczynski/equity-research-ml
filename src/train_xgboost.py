import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("Loading data...")
train = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_21d', 'return_63d', 'return_252d',
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    'volatility_21d', 'volatility_63d',
    'rsi_14', 'bb_position', 'volume_ratio_20d'
]

X_train = train[FEATURE_COLS].values
y_train = train['target'].values
X_test = test[FEATURE_COLS].values
y_test = test['target'].values

print(f"Training XGBoost on {len(X_train):,} rows...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

train_pred_proba = model.predict_proba(X_train)[:, 1]
test_pred_proba = model.predict_proba(X_test)[:, 1]
test_pred = model.predict(X_test)

print(f"\n=== XGBoost Results ===")
print(f"Train AUC: {roc_auc_score(y_train, train_pred_proba):.4f}")
print(f"Test AUC:  {roc_auc_score(y_test, test_pred_proba):.4f}")
print(f"\nTest set classification report:")
print(classification_report(y_test, test_pred, target_names=['Not top quintile', 'Top quintile']))

# Feature importance
print("=== Feature importance (XGBoost) ===")
importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.round(4))

# Save predictions for backtesting
test_with_preds = test.copy()
test_with_preds['prediction_proba'] = test_pred_proba
test_with_preds.to_parquet("data/test_predictions_xgboost.parquet", index=False)

# Save model
joblib.dump(model, "data/xgboost_model.pkl")
print("\n✅ Saved XGBoost model and predictions")

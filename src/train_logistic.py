import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("Loading data...")
train = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

# Define features (everything that's predictive — exclude metadata + target + leakage)
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

# Scaling matters for logistic regression (not for trees, but always good practice here)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training logistic regression on {len(X_train):,} rows...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Predictions
train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
test_pred = model.predict(X_test_scaled)

print(f"\n=== Logistic Regression Results ===")
print(f"Train AUC: {roc_auc_score(y_train, train_pred_proba):.4f}")
print(f"Test AUC:  {roc_auc_score(y_test, test_pred_proba):.4f}")
print(f"\nTest set classification report:")
print(classification_report(y_test, test_pred, target_names=['Not top quintile', 'Top quintile']))

# Coefficients — what features matter?
print("=== Feature importance (coefficients) ===")
coefs = pd.DataFrame({
    'feature': FEATURE_COLS,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(coefs.round(4))

# Save predictions for backtesting
test_with_preds = test.copy()
test_with_preds['prediction_proba'] = test_pred_proba
test_with_preds.to_parquet("data/test_predictions_logistic.parquet", index=False)

# Save the model
joblib.dump(model, "data/logistic_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")
print("\n✅ Saved model and predictions")
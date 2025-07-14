import pandas as pd
from io import StringIO
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import numpy as np
import warnings

# Suppress common warnings from sklearn for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


# --- Step 1: Load the Data ---

df = pd.read_csv("processed_keystroke_data.csv")

print("--- Data Loaded ---")
print(f"Shape of the dataset: {df.shape}")

# --- Step 2: Prepare Data for Modeling ---
X = df.drop('anxiety_score', axis=1)
y = df['anxiety_score']

# --- Step 3: Define Models, Cross-Validation, and RFE Configuration ---
# FIX: Changed n_splits from 5 to 3. With 9 samples, this creates 3 folds of 3 samples each,
# ensuring the test set always has more than 1 sample, allowing R2 to be calculated.
kf = KFold(n_splits=3, shuffle=True, random_state=42)

models_to_compare = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

rfe_estimators = {
    "Linear": LinearRegression(),
    "Tree": DecisionTreeRegressor(random_state=42)
}

# --- Step 4: Perform K-Fold Cross-Validation and Store Results ---
results = {}

def evaluate_model(model, X, y, kf, rfe_selector=None, n_features=None):
    """Evaluates a model using k-fold cross-validation, with optional RFE."""
    mse_scores, rmse_scores, r2_scores = [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if rfe_selector and n_features:
            rfe = RFE(estimator=rfe_selector, n_features_to_select=n_features, step=1)
            X_train_final = rfe.fit_transform(X_train_scaled, y_train)
            X_test_final = rfe.transform(X_test_scaled)
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))

    return np.mean(mse_scores), np.mean(rmse_scores), np.mean(r2_scores)

# Main evaluation loop
for model_name, model in models_to_compare.items():
    print(f"\n--- Evaluating {model_name} ---")

    avg_mse_no_rfe, avg_rmse_no_rfe, avg_r2_no_rfe = evaluate_model(model, X, y, kf)
    results[f"{model_name} (No RFE)"] = {
        "Avg MSE": avg_mse_no_rfe, "Avg RMSE": avg_rmse_no_rfe, "Avg R2": avg_r2_no_rfe
    }
    print(f"  Scores (No RFE): MSE={avg_mse_no_rfe:.4f}, RMSE={avg_rmse_no_rfe:.4f}, R2={avg_r2_no_rfe:.4f}")

    rfe_estimator_type = "Tree" if "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name else "Linear"
    rfe_estimator = rfe_estimators[rfe_estimator_type]
    print(f"  Using '{rfe_estimator_type}' RFE estimator.")

    best_rfe_performance = {"mse": float('inf'), "rmse": float('inf'), "r2": float('-inf'), "n_features": 0}

    max_features = X.shape[1]
    step = max(1, max_features // 5)
    feature_selection_range = range(step, max_features + 1, step)

    for n_features in feature_selection_range:
        avg_mse_rfe, avg_rmse_rfe, avg_r2_rfe = evaluate_model(model, X, y, kf, rfe_selector=rfe_estimator, n_features=n_features)

        # IMPROVEMENT: More robust check to handle potential NaN values.
        # A non-NaN R2 is always better than a NaN R2.
        current_best_r2 = best_rfe_performance["r2"]
        is_better = (not np.isnan(avg_r2_rfe) and (np.isnan(current_best_r2) or avg_r2_rfe > current_best_r2))

        if is_better:
            best_rfe_performance = {"mse": avg_mse_rfe, "rmse": avg_rmse_rfe, "r2": avg_r2_rfe, "n_features": n_features}

    best_n = best_rfe_performance['n_features']
    results[f"{model_name} (RFE {best_n} features)"] = {
        "Avg MSE": best_rfe_performance["mse"],
        "Avg RMSE": best_rfe_performance["rmse"],
        "Avg R2": best_rfe_performance["r2"]
    }
    print(f"  Best RFE: MSE={best_rfe_performance['mse']:.4f}, RMSE={best_rfe_performance['rmse']:.4f}, R2={best_rfe_performance['r2']:.4f} (found with {best_n} features)")


# --- Step 5: Display and Compare All Results ---
print("\n\n--- Comparison of All Model Configurations ---")
results_df = pd.DataFrame(results).T

print("\n--- Models Sorted by Average R2 (Highest First) ---")
print(results_df.sort_values(by="Avg R2", ascending=False))
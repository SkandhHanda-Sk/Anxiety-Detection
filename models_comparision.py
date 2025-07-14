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
data = """anxiety_score,dwell_time_mean,dwell_time_std,dwell_time_median,dwell_time_count,flight_time_mean,flight_time_std,flight_time_median,flight_time_count,inter_key_latency_mean,inter_key_latency_std,inter_key_latency_median,inter_key_latency_count,digraph_speed_mean,digraph_speed_std,digraph_speed_median,digraph_speed_count,trigraph_speed_mean,trigraph_speed_std,trigraph_speed_median,trigraph_speed_count,typing_speed_wpm,error_rate,error_key_count,pause_count,total_pause_duration_ms,capslock_count,enter_count,tab_count,shift_count
35,192.22222222222223,73.98945537308651,188.5,36,291.5458333318432,272.91758062414556,199.05000001192093,24,345.0342857156481,296.6390706844583,300.5,35,703.0499999821186,382.4500000178814,703.0499999821186,2,0.0,0.0,0.0,0,34.23764184152597,0.0,0,0,0.0,0,0,0,1
50,90.49725489476148,39.939699059666616,86.89999997615814,255,467.07960000753405,1157.106679172133,116.65000003576279,250,548.4350393700787,1150.6336325213338,206.3499999642372,254,434.461904769852,420.60738277857115,189.95000004768372,42,680.2444444629881,252.08498177054312,736.7999999523163,9,17.56302924946864,0.09019607843137255,23,11,54586.20000016689,0,0,0,2
67,4135.8229927000775,25219.26729863128,140.29999999701977,274,519.3913846161732,4298.131540348759,134.69999998807907,325,596.2076294277605,4048.8967197692036,258.90000000596046,367,313.5561403501452,245.5180202146435,272.59999999403954,57,502.5800000011921,83.88767251625448,481.9500000178814,10,16.444130381855995,0.035326086956521736,13,5,90950.09999996424,0,1,0,42
67,107.8000000004585,18.178897319808872,103.5,13,7940.741666665922,23693.486632383574,185.05000000447035,12,8045.024999999751,23693.657876881603,295.0000000074506,12,0.0,0.0,0.0,0,0.0,0.0,0.0,0,21.3465052854326,0.0,0,3,92982.59999999404,0,1,0,0
50,8534.289925373645,22706.89715200263,109.69999998807907,268,380.915725805346,908.2320469811281,129.04999999701977,248,421.2163333333532,861.3528173559904,202.29999999701977,300,374.48285714217593,436.3029167182636,181.5,35,698.3285714302745,347.9089442201415,559.4000000059605,7,16.230787240638847,0.24584717607973422,74,8,35687.899999946356,0,0,0,2
50,108.33846153846035,25.06558183055352,110.69999999995343,39,789.4868421052631,1054.9127771803946,433.3499999999767,38,897.6263157894749,1053.2175515468762,564.8999999999069,38,836.6600000000093,528.0684978295052,1053.0,5,1900.5,0.0,1900.5,1,9.466570442679053,0.15384615384615385,6,3,11040.0,0,0,0,0
41,262.8916666667598,259.6301037319592,98.20000000018626,12,2405.1636363633656,2893.7404740931424,976.9000000003725,11,1474.7368421052631,2502.3704545492415,208.69999999925494,19,0.0,0.0,0.0,0,0.0,0.0,0.0,0,22.632404906535474,0.85,17,5,24886.099999997765,0,0,0,0
62,63.42857142005648,34.82980925039282,59.749999940395355,28,155.4470588319442,342.40629138394,32.800000071525574,17,141.4592592627914,287.5257557094282,63.89999997615814,27,0.0,0.0,0.0,0,0.0,0.0,0.0,0,86.4508825188954,0.0,0,0,0.0,0,0,0,0
50,147.859558822368,44.31400634841533,142.30000001192093,272,237.88129251830432,264.29033848647435,160.6000000089407,294,377.48973509935746,266.2434754227292,291.45000000298023,302,376.31333333452545,289.2621287938199,280.55000001192093,30,1000.3200000047684,573.9802832849329,890.2999999821186,5,24.2809016518409,0.11221122112211221,34,1,2138.0,0,0,0,4
"""
df = pd.read_csv(StringIO(data))

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
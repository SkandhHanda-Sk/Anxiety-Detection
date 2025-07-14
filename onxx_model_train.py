import pandas as pd
import numpy as np
import os
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from typing import Tuple, Dict, Any

# --- Configuration Constants ---
DATA_FILE_PATH = 'processed_keystroke_data.csv'
ONNX_MODEL_PATH = 'anxiety_stacking_model_tuned.onnx'
TARGET_COLUMN = 'anxiety_score'
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5  # Using 5 for faster tuning, can be increased to 10
N_TUNING_TRIALS = 100 # Number of tuning iterations for Optuna

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads feature and target data from the specified CSV file.
    """
    print(f"Attempting to load data from '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The data file '{file_path}' was not found. "
            "Please ensure the file is in the correct directory."
        )
    
    df = pd.read_csv(file_path)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    print("Data loaded successfully.")
    return X, y

def save_model_as_onnx(pipeline: Pipeline, num_features: int, output_path: str):
    """
    Converts a scikit-learn pipeline to ONNX format and saves it.
    """
    print(f"\nConverting the final tuned pipeline to ONNX and saving to '{output_path}'...")
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Model successfully saved to '{output_path}'.")

def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: pd.Series) -> float:
    """
    The objective function for Optuna to optimize.
    It defines the hyperparameter search space, builds a model,
    and returns its cross-validated performance.
    """
    # --- Define Hyperparameter Search Space ---
    # 1. RandomForest Regressor
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
    rf_max_depth = trial.suggest_int('rf_max_depth', 5, 50)
    rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 10)

    # 2. Gradient Boosting Regressor
    gb_n_estimators = trial.suggest_int('gb_n_estimators', 50, 300)
    gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
    gb_max_depth = trial.suggest_int('gb_max_depth', 3, 10)

    # 3. Support Vector Regressor (SVR)
    svr_c = trial.suggest_float('svr_c', 0.1, 10.0, log=True)
    svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 1.0, log=True)

    # --- Build the Stacking Model with Suggested Hyperparameters ---
    estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            random_state=RANDOM_STATE
        )),
        ('svr', SVR(C=svr_c, epsilon=svr_epsilon, kernel='rbf'))
    ]

    # The meta-model can also be tuned, but RidgeCV is generally stable.
    # We will keep it fixed for this example.
    final_estimator = RidgeCV()

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )

    # --- Evaluate the Model using Cross-Validation ---
    k_folds = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = cross_val_score(
        stacking_model, X_train, y_train, cv=k_folds, 
        scoring='neg_mean_squared_error'
    )
    
    # Return the mean RMSE for this trial
    mean_rmse = np.mean(np.sqrt(-cv_scores))
    
    return mean_rmse

def main():
    """
    Main function to run the complete model tuning, training, and saving pipeline.
    """
    try:
        X, y = load_data(DATA_FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Split data before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale data after splitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Hyperparameter Tuning with Optuna ---
    print(f"\nStarting hyperparameter tuning with Optuna for {N_TUNING_TRIALS} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), n_trials=N_TUNING_TRIALS)

    print("\nHyperparameter tuning complete.")
    print(f"Best trial RMSE: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # --- Train Final Model with Best Hyperparameters ---
    print("\nTraining the final model with the best hyperparameters found...")
    
    # Retrieve best params
    best_params = study.best_params
    
    # Build the final model with the best params
    final_estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            min_samples_leaf=best_params['rf_min_samples_leaf'],
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=best_params['gb_n_estimators'],
            learning_rate=best_params['gb_learning_rate'],
            max_depth=best_params['gb_max_depth'],
            random_state=RANDOM_STATE
        )),
        ('svr', SVR(
            C=best_params['svr_c'], 
            epsilon=best_params['svr_epsilon'], 
            kernel='rbf'
        ))
    ]
    final_stacking_model = StackingRegressor(
        estimators=final_estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    final_stacking_model.fit(X_train_scaled, y_train)
    print("Final model training complete.")

    # --- Evaluate Final Tuned Model on Unseen Test Set ---
    y_pred = final_stacking_model.predict(X_test_scaled)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nFinal Tuned Model Performance on Unseen Test Set (RMSE): {final_rmse:.4f}")

    # --- Create and Save Final ONNX Pipeline ---
    deployment_pipeline = Pipeline([
        ('scaler', scaler), # Use the scaler that was fitted on the training data
        ('stacking_model', final_stacking_model)
    ])

    save_model_as_onnx(deployment_pipeline, X.shape[1], ONNX_MODEL_PATH)

if __name__ == '__main__':
    # To avoid issues with multiprocessing on some systems
    # and to see Optuna's progress bar correctly.
    main()

"""
Hyperparameter tuning complete.
Best trial RMSE: 12.8974
Best hyperparameters found:
  - rf_n_estimators: 59
  - rf_max_depth: 21
  - rf_min_samples_leaf: 2
  - gb_n_estimators: 185
  - gb_learning_rate: 0.24405036298683289
  - gb_max_depth: 8
  - svr_c: 0.4546137769160741
  - svr_epsilon: 0.5366832900668792

Training the final model with the best hyperparameters found...
Final model training complete.
"""
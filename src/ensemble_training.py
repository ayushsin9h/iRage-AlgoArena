import pandas as pd
import lightgbm as lgb
import gc
import time
from feature_engineering import reduce_mem_usage, engineer_features

# --- 1. File Paths ---
TRAIN_PATH = '../data/raw/train.parquet'
TEST_PATH = '../data/raw/test.parquet'
SUBMISSION_PATH = '../data/submissions/submission.csv'

def main():
    start_time = time.time()

    print("1. Loading datasets...")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    print("2. Aligning base features...")
    y_train = train['TARGET']
    test_ids = test['ID']

    base_features = [col for col in train.columns if col in test.columns and col != 'ID']
    X_train = train[base_features]
    X_test = test[base_features]

    del train, test
    gc.collect()

    print("3. Applying Feature Engineering...")
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Compress memory
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)

    print(f"Total predictive features generated: {X_train.shape[1]}")

    # --- 4. Initializing LightGBM Model ---
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 106,
        'max_depth': 13,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.97,
        'bagging_freq': 3,
        'min_child_samples': 84,
        'verbose': -1,
        'n_jobs': -1,  
        'random_state': 42
    }

    final_model = lgb.LGBMRegressor(**best_params, n_estimators=600)

    print("5. Training Master Alpha Model (CPU)...")
    final_model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    print("6. Generating Predictions...")
    predictions = final_model.predict(X_test)

    print("7. Creating submission file...")
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': predictions
    })

    submission.to_csv(SUBMISSION_PATH, index=False)

    execution_time = (time.time() - start_time) / 60
    print(f"SUCCESS! Pipeline completed perfectly in {execution_time:.2f} minutes.")

if __name__ == "__main__":
    main()
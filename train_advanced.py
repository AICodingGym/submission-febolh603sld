import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import time
import sys

print("Loading train data...")
train_df = pd.read_csv('data/extracted/train.csv')

print("Loading test data...")
test_df = pd.read_csv('data/extracted/test.csv')

print("Engineering features (u_in_cumsum, RC, u_in_R, u_in_C)...")
for df in [train_df, test_df]:
    df['u_in_cumsum'] = df.groupby('breath_id')['u_in'].cumsum()
    df['RC'] = df['R'] * df['C']
    df['u_in_R'] = df['u_in'] * df['R']
    df['u_in_C'] = df['u_in'] * df['C']

features = ['R', 'C', 'time_step', 'u_in', 'u_out', 'u_in_cumsum', 'RC', 'u_in_R', 'u_in_C']
target = 'pressure'

print("Splitting train/validation by breath_id (80/20)...")
unique_breath_ids = train_df['breath_id'].unique()
train_breath_ids, val_breath_ids = train_test_split(
    unique_breath_ids, test_size=0.2, random_state=42
)

train_split = train_df[train_df['breath_id'].isin(train_breath_ids)]
val_split = train_df[train_df['breath_id'].isin(val_breath_ids)]

# Filter training split to only rows where u_out == 0 before training
train_split_filtered = train_split[train_split['u_out'] == 0]

X_train = train_split_filtered[features]
y_train = train_split_filtered[target]

# Filter validation split for early stopping where u_out == 0
val_split_filtered = val_split[val_split['u_out'] == 0]
X_val = val_split_filtered[features]
y_val = val_split_filtered[target]

print(f"Train samples (u_out == 0): {len(X_train)}, Val samples (u_out == 0): {len(X_val)}")

print("Training LightGBM model with num_leaves=63, n_estimators=500, objective='mae'...")
model = lgb.LGBMRegressor(
    objective='mae',
    num_leaves=63,
    n_estimators=500,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)
print(f"Training took {time.time() - start_time:.2f} seconds.")

# Evaluate MAE on validation set where u_out == 0
val_preds = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"New Validation MAE (u_out == 0): {val_mae:.6f}")
sys.stdout.flush()

print("Training final model on all training data (filtered where u_out == 0)...")
train_full_filtered = train_df[train_df['u_out'] == 0]
X_full = train_full_filtered[features]
y_full = train_full_filtered[target]

final_model = lgb.LGBMRegressor(
    objective='mae',
    num_leaves=63,
    n_estimators=model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ else 500,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_full, y_full)

print("Predicting on test data...")
X_test = test_df[features]
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    'id': test_df['id'],
    'pressure': test_preds
})

submission.to_csv('submission.csv', index=False)
print("submission.csv successfully written with columns:", submission.columns.tolist())
print(submission.head())
sys.stdout.flush()

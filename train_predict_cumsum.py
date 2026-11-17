import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import time

print("Loading train data...")
train_df = pd.read_csv('data/extracted/train.csv')

print("Loading test data...")
test_df = pd.read_csv('data/extracted/test.csv')

print("Computing u_in_cumsum feature grouped by breath_id...")
train_df['u_in_cumsum'] = train_df.groupby('breath_id')['u_in'].cumsum()
test_df['u_in_cumsum'] = test_df.groupby('breath_id')['u_in'].cumsum()

features = ['R', 'C', 'time_step', 'u_in', 'u_out', 'u_in_cumsum']
target = 'pressure'

print("Splitting train/validation by breath_id (80/20)...")
unique_breath_ids = train_df['breath_id'].unique()
train_breath_ids, val_breath_ids = train_test_split(
    unique_breath_ids, test_size=0.2, random_state=42
)

train_split = train_df[train_df['breath_id'].isin(train_breath_ids)]
val_split = train_df[train_df['breath_id'].isin(val_breath_ids)]

X_train = train_split[features]
y_train = train_split[target]
X_val = val_split[features]
y_val = val_split[target]

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

print("Training LightGBM model with u_in_cumsum feature...")
model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)
print(f"Training took {time.time() - start_time:.2f} seconds.")

# Evaluate MAE on validation set where u_out == 0
val_preds = model.predict(X_val)
u_out_mask = (X_val['u_out'] == 0)
val_mae = mean_absolute_error(y_val[u_out_mask], val_preds[u_out_mask])
print(f"New Validation MAE (u_out == 0): {val_mae:.6f}")

print("Training final model on all training data...")
X_full = train_df[features]
y_full = train_df[target]

final_model = lgb.LGBMRegressor(
    n_estimators=model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ else 1000,
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
print("submission.csv updated successfully with columns:", submission.columns.tolist())
print(submission.head())

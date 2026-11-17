import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

train_df = pd.read_csv('data/extracted/train.csv')
train_df['u_in_cumsum'] = train_df.groupby('breath_id')['u_in'].cumsum()

features = ['R', 'C', 'time_step', 'u_in', 'u_out', 'u_in_cumsum']
target = 'pressure'

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

model = lgb.LGBMRegressor(
    objective='mae',
    n_estimators=100, # quick check
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
u_out_mask = (X_val['u_out'] == 0)
val_mae = mean_absolute_error(y_val[u_out_mask], val_preds[u_out_mask])
print(f"Validation MAE (u_out == 0) [100 trees]: {val_mae:.6f}")

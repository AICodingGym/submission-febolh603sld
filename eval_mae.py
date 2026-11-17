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

val_split = train_df[train_df['breath_id'].isin(val_breath_ids)]
X_val = val_split[features]
y_val = val_split[target]

# We can load or test model, or train a quick model / evaluate
print("Validation set loaded. u_out==0 proportion:", (X_val['u_out'] == 0).mean())

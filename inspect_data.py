import pandas as pd
import numpy as np

train_df = pd.read_csv('data/extracted/train.csv', nrows=1000)
print("Train columns:", train_df.columns.tolist())
print("Unique breath_ids in sample:", train_df['breath_id'].nunique())
print(train_df.head())

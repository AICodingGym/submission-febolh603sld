import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Let's inspect what val_mae was or quickly calculate it
train_df = pd.read_csv('data/extracted/train.csv', nrows=100000)
print("Quick check script loaded.")

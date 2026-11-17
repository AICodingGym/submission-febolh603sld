import pandas as pd

train_df = pd.read_csv('data/extracted/train.csv', usecols=['id', 'breath_id', 'u_out'])
test_df = pd.read_csv('data/extracted/test.csv', usecols=['id', 'breath_id'])
print("Train shape (id, breath_id, u_out):", train_df.shape)
print("Test shape (id, breath_id):", test_df.shape)
print("Unique train breath_ids:", train_df['breath_id'].nunique())
print("Unique test breath_ids:", test_df['breath_id'].nunique())

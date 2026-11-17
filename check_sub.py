import pandas as pd
sub = pd.read_csv('submission.csv')
print("Submission shape:", sub.shape)
print("Columns:", sub.columns.tolist())
print(sub.head())
print(sub.tail())

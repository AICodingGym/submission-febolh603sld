import zipfile
import os

os.makedirs('data/extracted', exist_ok=True)
with zipfile.ZipFile('data/ventilator-pressure-prediction.zip', 'r') as z:
    z.extractall('data/extracted')
print("Extracted files:", os.listdir('data/extracted'))

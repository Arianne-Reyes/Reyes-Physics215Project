import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Assuming 'data/raw' folder already exists, so no need to create it here
df.to_csv("data/raw/diabetes_raw.csv", index=False)

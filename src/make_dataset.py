import pandas as pd
from sklearn.datasets import load_diabetes
import os

def save_raw_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/diabetes_raw.csv", index=False)

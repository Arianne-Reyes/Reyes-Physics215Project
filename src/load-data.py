import pandas as pd
import os
from sklearn.datasets import load_diabetes

def load_and_save_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save raw data
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/diabetes_raw.csv", index=False)

    return df

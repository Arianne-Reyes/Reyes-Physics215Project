import pandas as pd
from sklearn.datasets import load_diabetes

def load_diabetes_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df, X, y, feature_names

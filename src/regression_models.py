from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np

def run_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "LASSO": Lasso(alpha=0.1),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        results[name] = -np.mean(scores)

    return results

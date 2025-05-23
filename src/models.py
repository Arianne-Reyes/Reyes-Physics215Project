from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

def get_models():
    return {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "LASSO": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

def evaluate_models(models, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    predictions = []
    labels = []

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=kf)
        mse = mean_squared_error(y, y_pred)
        predictions.append(y_pred)
        labels.append(f"{name} (MSE={mse:.1f})")
    return predictions, labels

def corrupt_target(y, index, offset):
    y_corrupted = y.copy()
    y_corrupted[index] += offset
    return y_corrupted

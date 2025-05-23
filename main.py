from src.load_data import load_and_save_data
from src.visualize import scatter_bmi_vs_target
from src.models import evaluate_models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold

# Load and save data
df = load_and_save_data()
X = df.drop(columns="target").values
y = df["target"].values

# Visualize and save scatter plot
scatter_bmi_vs_target(X, y)

# Define models
models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "LASSO": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate and save results
evaluate_models(models, X, y, kf)

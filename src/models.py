import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

def evaluate_models(models, X, y, kf):
    results = []

    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=kf)
        mse = mean_squared_error(y, y_pred)
        results.append({"Model": name, "MSE": mse})

        plt.scatter(y, y_pred, label=f"{name} (MSE={mse:.1f})", alpha=0.6)

    # Plot reference line
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("Actual vs Predicted Values for Regression Models")
    plt.legend()
    plt.grid(True)

    os.makedirs("data/final", exist_ok=True)
    plt.savefig("data/final/model_predictions.png")
    plt.close()

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(results).to_csv("data/processed/model_performance.csv", index=False)

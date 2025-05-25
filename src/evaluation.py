import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title} (MSE: {mse:.2f})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_feature_vs_target(X_feature, y, xlabel):
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(X_feature, y, c=y, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label="Target (Disease Progression)")
    plt.xlabel(xlabel)
    plt.ylabel("Disease Progression (1 year later)")
    plt.title(f"{xlabel} vs Disease Progression")
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted(y_true, y_preds, labels):
    plt.figure(figsize=(10, 6))
    for y_pred, label in zip(y_preds, labels):
        plt.scatter(y_true, y_pred, alpha=0.6, label=label)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_with_corruption(y_corrupted, y_pred_corrupted, corrupted_index):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_corrupted, y_pred_corrupted, alpha=0.6, label="With Corrupted Point")
    plt.plot([y_corrupted.min(), y_corrupted.max()],
             [y_corrupted.min(), y_corrupted.max()], 'k--', lw=2)
    plt.scatter(y_corrupted[corrupted_index], y_pred_corrupted[corrupted_index],
                color='red', s=100, edgecolor='black', label='Corrupted Point')
    plt.annotate("Corrupted", xy=(y_corrupted[corrupted_index], y_pred_corrupted[corrupted_index]),
                 xytext=(y_corrupted[corrupted_index] + 20, y_pred_corrupted[corrupted_index] - 10),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.xlabel("Actual Target (with corruption)")
    plt.ylabel("Predicted Target")
    plt.title("Effect of Corrupted Data Point")
    plt.legend()
    plt.grid(True)
    plt.show()

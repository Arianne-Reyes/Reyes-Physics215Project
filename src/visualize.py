import os
import matplotlib.pyplot as plt

def scatter_bmi_vs_target(X, y, feature_index=2):
    X_feature = X[:, feature_index]

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(X_feature, y, c=y, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label="Target (Disease Progression)")
    plt.xlabel("BMI (standardized)")
    plt.ylabel("Disease Progression (1 year later)")
    plt.title("Scatter Plot of BMI vs Disease Progression")
    plt.grid(True)

    os.makedirs("data/final", exist_ok=True)
    plt.savefig("data/final/bmi_vs_progression.png")
    plt.close()

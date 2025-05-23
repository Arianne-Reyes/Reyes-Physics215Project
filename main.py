from src.load_data import load_diabetes_data
from src.visualize import plot_feature_vs_target, plot_actual_vs_predicted, plot_with_corruption
from src.models import get_models, evaluate_models, corrupt_target

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

def main():
    df, X, y, feature_names = load_diabetes_data()
    print("Data shape:", df.shape)

    feature_index = 2
    plot_feature_vs_target(X[:, feature_index], y, xlabel=feature_names[feature_index])

    models = get_models()
    predictions, labels = evaluate_models(models, X, y)
    plot_actual_vs_predicted(y, predictions, labels)

    corrupted_index = 100
    y_corrupted = corrupt_target(y, corrupted_index, 200)
    model = LinearRegression()
    y_pred_corrupted = cross_val_predict(model, X, y_corrupted, cv=5)
    plot_with_corruption(y_corrupted, y_pred_corrupted, corrupted_index)

if __name__ == "__main__":
    main()

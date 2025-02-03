'''
Model Evaluation Script

This script loads the trained model, evaluates it on the cleaned dataset, and generates evaluation metrics.
It also produces visualizations for actual vs. predicted values and residual plots.

Author: Arnav Bajaj
Repository: Car-Price-Predictor
'''

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_model(filepath):
    """Loads the trained model from file."""
    return joblib.load(filepath)

def load_data(filepath):
    """Loads the dataset for evaluation."""
    return pd.read_csv(filepath)

def evaluate_model(y_true, y_pred):
    """Calculates and prints model evaluation metrics."""
    print("Model Evaluation Metrics:")
    print(f"MAE  (Mean Absolute Error): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE  (Mean Squared Error): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

def plot_actual_vs_predicted(y_true, y_pred):
    """Generates scatter plot for actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", color="red")  # Ideal line
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Car Prices")
    plt.show()

def plot_residuals(y_pred, residuals):
    """Generates residual plot."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

def save_results(y_true, y_pred, output_filepath):
    """Saves model evaluation results to a CSV file."""
    results = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2 Score": r2_score(y_true, y_pred)
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_filepath, index=False)
    print("Evaluation results saved!")

if __name__ == "__main__":
    # Load model and dataset
    model_path = 'models/model.pkl'
    data_path = 'data/Cleaned_Car_Price_Data.csv'
    model = load_model(model_path)
    df = load_data(data_path)
    
    # Prepare features and target
    X = df.drop(columns=["price"])
    y = df["price"]
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Evaluate model
    evaluate_model(y, y_pred)
    
    # Generate plots
    plot_actual_vs_predicted(y, y_pred)
    residuals = y - y_pred
    plot_residuals(y_pred, residuals)
    
    # Save results
    save_results(y, y_pred, "data/model_evaluation_results.csv")

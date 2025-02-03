'''
Model Training Script

This script loads the cleaned car price dataset, preprocesses it, and trains multiple machine learning models using GridSearchCV.
The best performing model is then saved for future predictions.

Author: Arnav Bajaj
Repository: Car-Price-Predictor
'''

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def load_data(filepath):
    """Loads dataset from the given file path."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Splits dataset into features and target variable."""
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def build_pipeline():
    """Constructs the preprocessing and model training pipeline."""
    categories = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 
                  'enginetype', 'fuelsystem', 'cylindernumber', 'doornumber']
    numericals = [col for col in X.columns if col not in categories]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', QuantileTransformer(n_quantiles=200), numericals),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def define_param_grid():
    """Defines hyperparameter grid for GridSearchCV."""
    return [
        {"regressor": [ElasticNet()], "regressor__alpha": np.logspace(-3, 3, 13), "regressor__l1_ratio": [0.1, 0.5, 0.9]},
        {"regressor": [RandomForestRegressor()], "regressor__n_estimators": [50, 100, 200], "regressor__max_depth": [None, 10, 20], "regressor__min_samples_split": [2, 5, 10]},
        {"regressor": [GradientBoostingRegressor()], "regressor__n_estimators": [50, 100, 200], "regressor__learning_rate": [0.01, 0.1, 0.2], "regressor__max_depth": [3, 5, 10]},
        {"regressor": [SVR()], "regressor__kernel": ["linear", "rbf"], "regressor__C": np.logspace(-2, 2, 5), "regressor__epsilon": [0.01, 0.1, 0.5]},
        {"regressor": [LinearSVR(max_iter=10000)], "regressor__C": np.logspace(-2, 2, 5), "regressor__epsilon": [0.01, 0.1, 0.5]},
        {"regressor": [KNeighborsRegressor()], "regressor__n_neighbors": [3, 5, 10], "regressor__weights": ["uniform", "distance"], "regressor__p": [1, 2]},
        {"regressor": [Ridge()], "regressor__alpha": np.logspace(-5, 5, 13)},
        {"regressor": [Lasso()], "regressor__alpha": np.logspace(-5, 5, 13)}
    ]

def train_model(X, y, pipeline, param_grid):
    """Trains model using GridSearchCV and returns the best estimator."""
    model = GridSearchCV(pipeline, param_grid, cv=7, n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_

def save_model(model, filepath):
    """Saves the trained model to a file."""
    joblib.dump(model, filepath)
    print("Model saved successfully!")

if __name__ == "__main__":
    # Load dataset
    data_path = 'data/Cleaned_Car_Price_Data.csv'
    df = load_data(data_path)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Build pipeline
    pipeline = build_pipeline()
    
    # Define parameter grid
    param_grid = define_param_grid()
    
    # Train model
    best_model = train_model(X, y, pipeline, param_grid)
    
    # Save best model
    save_model(best_model, "models/model.pkl")
'''
Data Processing Script

This script performs Exploratory Data Analysis (EDA) on the Car Price dataset.
It loads the dataset, displays summary statistics, and visualizes key relationships.
Additionally, it cleans and standardizes string values before saving the processed dataset.

Author: Arnav Bajaj
Repository: Car-Price-Predictor
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Loads dataset from the given file path."""
    return pd.read_csv(filepath)

def basic_info(df):
    """Displays basic dataset information."""
    print(f"Dataset Shape: {df.shape}")
    df.info()
    print(df.describe())

def check_missing_values(df):
    """Checks for missing values in the dataset."""
    print("Missing Values:", df.isnull().sum().values.sum())

def standardize_strings(df, columns):
    """Standardizes string values by converting to lowercase and stripping spaces."""
    for col in columns:
        df[col] = df[col].str.lower().str.strip()

def visualize_data(df):
    """Generates visualizations for data exploration."""
    # Histogram for numerical features
    df.hist(figsize=(24, 16), bins=30)
    plt.show()
    
    # Count plots for categorical features
    categorical_columns = ["fueltype", "carbody", "doornumber", "drivewheel", 
                           "enginelocation", "enginetype", "fuelsystem", "aspiration", "CarName",
                           "cylindernumber", "doornumber"]
    
    fig, axes = plt.subplots(4, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_columns):
        sns.countplot(y=df[col], ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    plt.show()

def clean_data(df):
    """Drops unnecessary columns and saves the cleaned dataset."""
    df.drop(columns=["car_ID", "CarName", "enginelocation"], inplace=True)
    df.to_csv("data/Cleaned_Car_Price_Data.csv", index=False)
    print("Cleaned dataset saved.")

if __name__ == "__main__":
    # File path to dataset
    data_path = "Data/Car_Price_Data.csv"
    
    # Load dataset
    df = load_data(data_path)
    print(df.head())
    
    # Display dataset info
    basic_info(df)
    
    # Check for missing values
    check_missing_values(df)
    
    # Standardize string columns
    string_columns = ["CarName", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel", 
                      "enginelocation", "fuelsystem", "enginetype", "cylindernumber", "doornumber"]
    standardize_strings(df, string_columns)
    
    # Visualize data
    visualize_data(df)
    
    # Clean and save processed data
    clean_data(df)
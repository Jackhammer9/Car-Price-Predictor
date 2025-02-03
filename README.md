<div align = 'center'> <img src= "https://raw.githubusercontent.com/Jackhammer9/Car-Price-Predictor/refs/heads/main/logo.webp" height=250px width=250px> </div>

A machine learning project that predicts used car prices based on various features such as make, model, year, engine capacity, and more.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/Follow%20on-LinkedIn-blue.svg)](https://www.linkedin.com/in/arnav-bajaj/)

--------------------------------------------------------------------------------
Table of Contents
-----------------
1. Overview  
2. Dataset  
3. Project Structure  
4. Getting Started  
5. EDA (Exploratory Data Analysis)  
6. Modeling Approach  
7. Results  
8. How to Use  
9. Future Improvements  
10. Contributing  
11. License

--------------------------------------------------------------------------------
Overview
--------
Predicting the price of used cars is a valuable exercise for both car dealerships
and individual car owners. By analyzing historical data, we can use machine
learning techniques to estimate the market value of a used car based on features
like make, model, mileage, engine capacity, fuel type, and more.

This project showcases:
- End-to-end data handling (cleaning, feature engineering)
- Multiple regression algorithms (e.g., Linear Regression, Random Forest, XGBoost)
- Model tuning and evaluation

Goal:
Create a predictive model that accurately estimates a car’s price given its
attributes.

--------------------------------------------------------------------------------
Dataset
-------
Source: Car Price Prediction Dataset by Hellbuoy on Kaggle
(https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)

This dataset contains various features like:
- Car name (brand/model)
- Year of manufacture
- Selling price
- Present price (original price)
- Kilometers driven
- Fuel type, Seller type, Transmission
- And more…

Note: Please check the dataset’s licensing and usage permissions before
commercial use.

--------------------------------------------------------------------------------

- data/: Contains raw or preprocessed data (or a README with a link to the dataset).
- notebooks/: Jupyter notebooks for EDA, model training, and experiments.
- src/: Python scripts for data preprocessing, modeling, etc.
- models/: Serialized model files for quick loading.

--------------------------------------------------------------------------------
Getting Started
---------------
1. Clone the Repository

   ```
   git clone https://github.com/Jackhammer9/Car-Price-Predictor.git
   cd Car-Price-Predictor
   ```

2. Create a Virtual Environment (Optional but Recommended)

   # Using conda
   ```
   conda create -n car-price-predictor python=3.8
   conda activate car-price-predictor
   ```

   # or using venv
   ```
   python -m venv env
   source env/bin/activate
   ```

4. Install Dependencies
   ```
   pip install -r requirements.txt
   ```

5. Download the Dataset
   If not included, download the dataset from Kaggle:
   https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
   and place it in the data/ folder (e.g., Car-Data.csv).

--------------------------------------------------------------------------------
EDA (Exploratory Data Analysis)
-------------------------------
During EDA, we examine:
- Missing values and possible imputation strategies
- Distribution of numeric variables (mileage, price, etc.)
- Categorical variable analysis (fuel type, seller type, etc.)
- Correlation between features and target (selling price)

--------------------------------------------------------------------------------
Modeling Approach
-----------------
We tried multiple algorithms to find the best performer:

1. Linear Regression
   - Pros: Interpretable, fast to train
   - Cons: May not capture nonlinear relationships well

2. Random Forest
   - Pros: Handles nonlinearities, robust to outliers, can measure feature importance
   - Cons: Can be slower, may overfit if not tuned properly

3. XGBoost
   - Pros: Often achieves high accuracy on tabular data, can handle missing data well
   - Cons: Tuning can be more involved

Hyperparameter Tuning:
We used GridSearchCV or RandomizedSearchCV for each model to find optimal
parameters (e.g., max depth, n_estimators, learning rate).

--------------------------------------------------------------------------------
Future Improvements
-------------------
1. Advanced Feature Engineering:
   - Derived features like car age, brand-specific average prices, etc.

2. Ensemble Methods:
   - Combine multiple models (e.g., stacking) for improved performance.

3. Deep Learning:
   - Experiment with neural networks on tabular data (though benefits may vary).

4. Deployment:
   - Containerize with Docker or deploy to AWS/Azure/GCP.

--------------------------------------------------------------------------------
Contributing
------------
Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request, or open an issue.

--------------------------------------------------------------------------------
License
-------
Distributed under the MIT License. See LICENSE for more information.

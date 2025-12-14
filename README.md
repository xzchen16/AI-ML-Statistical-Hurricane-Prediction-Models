Project Overview

This workspace contains code, data, and outputs for hurricane prediction modeling using multiple approaches (ANN, KNN, Lasso, and XGBoost). The dataset is stored in R binary format (.Rdata), model code is primarily in Python (.py) and R (.R), and results are saved as NumPy arrays (.npy) and RData files.

Folder Structure
- Code/
  - ANN.py: Python script to train/evaluate an Artificial Neural Network model.
  - KNN.py: Python script to train/evaluate a K-Nearest Neighbors model.
  - lasso.R: R script to train/evaluate a Lasso regression model.
  - Results_Comparison.py: Python script to compare model outputs across approaches.
  - XGBoost.py: Python script to train/evaluate an XGBoost model.
- Data/
  - independentvariable.Rdata: RData containing predictor (feature) matrix.
  - targetvariable.Rdata: RData containing target variable vector
- Output/
  - ann_results.npy: NumPy array with ANN prediction results.
  - climat_results.npy: NumPy array with climatology baseline results.
  - knn_results.npy: NumPy array with KNN predictions results.
  - lassoresults.Rdata: RData containing Lasso model results.
  - xgb_results.npy: NumPy array with XGBoost results.

Environment and Dependencies
- Python: numpy, pandas, scikit-learn, xgboost (as applicable), matplotlib/seaborn for visualization; optionally `pyreadr` to load RData.
- R: base R plus packages used by lasso.R (e.g., glmnet).

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 20:23:30 2025
"""

import rdata
import pandas as pd
import numpy as np
import os

#read in predictor and targets
feature_dataset = rdata.read_rda("../Data/independentvariable.Rdata")
target_dataset = rdata.read_rda("../Data/targetvariable.Rdata")
features = feature_dataset['dataX']
targets = target_dataset['yy0']

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# XGBoost
# test best hyperparams for each i each j at each year (sliding window CV)
xgb_results = []
xgb_best_params = []

# ---- hyperparameter grid (kept small for speed on tiny yearly windows) ----
param_grid = {
    "objective": ["reg:squarederror"],   # use 'reg:squarederror' for non-count targets
    "max_depth": [3, 5,10],
    "n_estimators": [50, 100, 200],
    "subsample": [0.7],
    "min_child_weight": [1, 3, 5],
}



for i in range(len(targets)):
    print("i = ", i)
    xgb_results_i = []
    xgb_best_params_i = []
    for j in range(len(features)):
        print("j = ", j)
        xgb_results_ij = []
        xgb_best_params_ij = []
        for year in range(30, 74):  # train on [year-30, ..., year-1], test on [year]
            print(year)
            x_train = features[j].values[year-30:year, :]
            y_train = targets[i][year-30:year]
            x_test  = features[j].values[year, :]

            # model + grid search
            model = XGBRegressor(
                tree_method="hist",      # fast, good default
                random_state=42,
                n_jobs=-1
            )
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="neg_mean_absolute_error",  # or 'neg_poisson_deviance' (sklearn>=1.4) if using Poisson
                n_jobs=-1,
                refit=True
            )
            grid.fit(x_train, y_train)

            best = grid.best_estimator_
            pred = best.predict(x_test.reshape(1, -1))[0]

            xgb_results_ij.append(pred)
            xgb_best_params_ij.append(grid.best_params_)

        xgb_results_i.append(xgb_results_ij)
        xgb_best_params_i.append(xgb_best_params_ij)
    xgb_results.append(xgb_results_i)
    xgb_best_params.append(xgb_best_params_i)
    
np.save('Outputs/xgb_results.npy',xgb_results)

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 22:16:36 2025
"""

import rdata
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from scipy.stats import poisson

#read in predictor and targets
feature_dataset = rdata.read_rda("../Data/independentvariable.Rdata")
target_dataset = rdata.read_rda("../Data/targetvariable.Rdata")
features = feature_dataset['dataX']
targets = target_dataset['yy0']

#knn
#test best_k for each i each j at each year(swcv)
knn_results = []
for i in range(len(targets)):
  knn_results_i = []
  for j in range(len(features)):
    knn_results_ij = []
    knn_h_ij = []
    for year in range(30, 74):
      x_train = features[j].values[year-30:year,:]
      y_train = targets[i][year-30:year]
      x_test = features[j].values[year,:]

      #find best k for knn model
      knn = KNeighborsRegressor()
      param_grid = {'n_neighbors': np.arange(1, 10)}
      grid_search = GridSearchCV(knn, param_grid, cv=5)
      grid_search.fit(x_train, y_train)
      best_k = grid_search.best_params_
      knn = KNeighborsRegressor(n_neighbors=best_k['n_neighbors'])
      knn.fit(x_train, y_train)
      pred_knn = knn.predict(x_test.reshape(1, -1))[0]

      knn_results_ij.append(pred_knn)
    knn_results_i.append(knn_results_ij)
  knn_results.append(knn_results_i)

np.save('Outputs/knn_results.npy',knn_results)
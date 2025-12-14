# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 20:33:15 2025
"""
import rdata
import pandas as pd
import numpy as np
import os
from scipy.stats import poisson

#read in predictor and targets
feature_dataset = rdata.read_rda("../Data/independentvariable.Rdata")
target_dataset = rdata.read_rda("../Data/targetvariable.Rdata")
features = feature_dataset['dataX']
targets = target_dataset['yy0']

obs_results = np.array([targets]*9)
obs_results = np.swapaxes(obs_results, 0, 1)[:,:,30:]
target_var_list = ['ATTC', 'ATHU', 'ATMH', 'CATC', 'CAHU', 'CAMH', 'GUTC', 'GUHU','GUMH']

#define H score function
def get_H_score(pred, obs, climat_pred):
    # for 0 or negative predictions change to a small number to make log valid
    pred[pred <= 0] = 1e-12
    # Negative log-likelihood under model prediction
    score = -poisson.logpmf(obs, pred)

    # Negative log-likelihood under climatology (average)
    clima1 = -poisson.logpmf(obs, climat_pred)

    # Skill scores
    H1 = clima1 - score

    # delta H
    return H1

def get_RMSE(pred, obs):
    rmse = np.sqrt(np.mean((pred - obs) ** 2, axis=2))
    return rmse

#read results
reg_data = rdata.read_rda('../Output/lassoresults.Rdata')
reg_data = reg_data['allresults']
reg_results = np.zeros((9,9,44))
for i in range(9):
    for j in range(9):
        for year in range(44):
            reg_results[i,j,year] = reg_data[0][i][j]['resSWCV'].iloc[year,0]
            
reg_h_ori =  np.zeros((9,9,44))
for i in range(9):
    for j in range(9):
        for year in range(44):
            reg_h_ori[i,j,year] = reg_data[0][i][j]['tabSWCV'][year,0].values

reg_h_ori_mean = np.mean(reg_h_ori,axis=2)

cli_results = np.load('../Outputs/climat_results.npy') #climatology results
ann_results = np.load('../Outputs/ann_results.npy') #ANN results
knn_results = np.load('../Outputs/knn_results.npy') #KNN results
xgb_results = np.load('../Outputs/xgb_results.npy') #XGB results

reg_h = get_H_score(reg_results, obs_results, cli_results)
reg_h_mean = np.mean(reg_h,axis=2)

ann_h = get_H_score(ann_results, obs_results, cli_results)
ann_h_mean = np.mean(ann_h,axis=2)

knn_h = get_H_score(knn_results, obs_results, cli_results)
knn_h_mean = np.mean(knn_h,axis=2)

xgb_h = get_H_score(xgb_results, obs_results, cli_results)
xgb_h_mean = np.mean(xgb_h,axis=2)

#draw prediction and truth graph
import matplotlib.pyplot as plt 
#get best pred from climatology
cli_best = cli_results[:,0,:]
obs_record = obs_results[:,0,:]

    
def create_pred_curve(pred_results, pred_h_mean, cli, obs, model_name): 
    #cli = climatology results
    pred_best_idx = np.argmax(pred_h_mean, axis=1)
    pred_best = np.zeros([9,44])
    for i in range(9):
        pred_best[i,:] = pred_results[i, pred_best_idx[i], :]

    # ===== 3x3 subplots for all 9 series =====
    start_year = 1981  # adjust if needed
    years = np.arange(start_year, start_year + obs.shape[1])


    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey='row')
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.plot(years, obs[i, :], label='Observed', linewidth=1)
        ax.plot(years, cli[i, :],  label='Climatology', linewidth=1)
        ax.plot(years, pred_best[i, :], label= model_name, linewidth=1)

        ax.set_title(f'{target_var_list[i]}', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i % 3 == 0:
            ax.set_ylabel('Hurricane Count')
        if i >= 6:
            ax.set_xlabel('Year')

    # Add a single legend for the entire figure
    fig.legend(['Observed', 'Climatology', f'Best {model_name} Model'], loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.993))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'Observed vs Climatology vs {model_name} Hurricane Counts', fontsize=14, y=1.01)
    plt.show()
    
create_pred_curve(reg_results,reg_h_mean,cli_best,obs_record,'Lasso')
create_pred_curve(knn_results,knn_h_mean,cli_best,obs_record,'KNN')   
create_pred_curve(ann_results,ann_h_mean,cli_best,obs_record,'ANN')   
create_pred_curve(xgb_results,xgb_h_mean,cli_best,obs_record,'XGBoost') 

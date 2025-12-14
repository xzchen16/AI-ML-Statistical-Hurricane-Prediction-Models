# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 08:33:32 2025
"""

import rdata
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

#read in predictor and targets
feature_dataset = rdata.read_rda("../Data/independentvariable.Rdata")
target_dataset = rdata.read_rda("../Data/targetvariable.Rdata")
features = feature_dataset['dataX']
targets = target_dataset['yy0']

class ann_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ann_model, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)  # must be integer
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim // 2, 1)  # capital 'L'
        
        # # Initialize weights using He initialization for ReLU activations
        # init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        # init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        # init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        
        # init.constant_(self.linear1.weight, 0)
        # init.constant_(self.linear2.weight, 0)
        # init.constant_(self.linear2.weight, 0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1,1)  # (N,1) for regression
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


criterion = nn.MSELoss()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
def test(model, device, test_loader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss.append(criterion(output, y).item())
    test_loss = np.mean(test_loss)
    #print(f'\nTest set: Average loss: {test_loss:.4f}')
    return output
    
# Parameters
# Parameters for the model
parser = argparse.ArgumentParser(description='ANN Hurricane Predictor')
parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}

if use_cuda:
    cuda_kwargs = {'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
            
ann_results=[]
for i in range(len(targets)):
    print("i = ", i)
    ann_results_i = []
    
    for j in range(len(features)):
        print("j = ", j)
        ann_results_ij = []
        for year in range(30, 74):  # train on [year-30, ..., year-1], test on [year]
            print(year)
            x_train = features[j].values[year-30:year, :]
            y_train = targets[i][year-30:year]
            x_test  = features[j].values[year:year+1, :]
            y_test = targets[i][year:year+1]
            train_dataset = WeatherDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, **train_kwargs)
            test_dataset = WeatherDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, **train_kwargs)

            # model
            model = ann_model(input_dim = x_train.shape[1], hidden_dim = 100).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
            
            for epoch in range(1, args.epochs + 1):
                #print(f"Starting Epoch {epoch}/{args.epochs}")
                train(args, model, device, train_loader, optimizer, epoch)
                test(model, device, test_loader)
                scheduler.step()

            pred = test(model, device, test_loader).to('cpu').numpy()[0,0]
            ann_results_ij.append(pred)
        ann_results_i.append(ann_results_ij)
    ann_results.append(ann_results_i)
        
np.save('Outputs/ann_results.npy',ann_results)


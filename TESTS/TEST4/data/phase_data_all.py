import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

import os
file_path = os.path.dirname(os.path.abspath(__file__))


datasets = [("../final/bike.npz", "bike"),
            ("../final/elevators.npz", "elevators"),
            ("../final/poletele.npz", "poletele"),
            ("../final/road3d.npz", "road3d")]

for dataset in datasets:
   
   if not os.path.exists(dataset[0]):
      raise ValueError("Dataset not found. Please run parse_data.py first.")

   data = np.load(dataset[0])
   X_train = data['train_x']
   Y_train = data['train_y']
   X_test = data['test_x']
   Y_test = data['test_y']

   Ntrain = X_train.shape[1]
   Ntest = X_test.shape[1]
   Dim = X_train.shape[0]

   with open(file_path + '/' + dataset[1] + '.train.feature', 'w') as f:
      f.write(str(Ntrain) + ' ' + str(Dim) + '\n')
      for i in range(Dim):
         for j in range(Ntrain):
            f.write(str(X_train[i, j]) + ' ')
         f.write('\n')

   with open(file_path + '/' + dataset[1] + '.train.label', 'w') as f:
      f.write(str(Ntrain) + '\n')
      for i in range(Ntrain):
         f.write(str(Y_train[i]) + '\n')
         
   with open(file_path + '/' + dataset[1] + '.test.feature', 'w') as f:
      f.write(str(Ntest) + ' ' + str(Dim) + '\n')
      for i in range(Dim):
         for j in range(Ntest):
            f.write(str(X_test[i, j]) + ' ')
         f.write('\n')

   with open(file_path + '/' + dataset[1] + '.test.label', 'w') as f:
      f.write(str(Ntest) + '\n')
      for i in range(Ntest):
         f.write(str(Y_test[i]) + '\n')

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('FuelConsumption.csv - FuelConsumption.csv.csv')

#dataset['experience'].fillna(0, inplace=True)

#dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

 

#Converting words to integer values
#def convert_to_int(word):
  #  word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
  #              'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
   # return word_dict[word]

#X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))
cdf = dataset[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
msk = np.random.rand(len(dataset)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)

#Fitting model with trainig data
# regressor.fit(X, y)

# Saving model to disk
pickle.dump(regr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
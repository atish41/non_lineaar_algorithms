# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:46:22 2023

@author: ATISHKUMAR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\jule\18th jule\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')

#seprating the dataset into independant and dependant variable

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting svr to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='poly', degree=4, gamma='auto')
regressor.fit (x,y)

#predicting the new results
y_pred=regressor.predict([[6.5]])
#this is not good model becoz we get130k

#visulizing the svr 
plt.scatter(x, y, color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('positon salary')
plt.ylabel('salary')
plt.show()
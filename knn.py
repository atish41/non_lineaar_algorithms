# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:39:09 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\jule\18th jule\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=4,p=1)
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])

plt.scatter(x, y, color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff(KNN)')
plt.xlabel('position')
plt.ylabel('salary')
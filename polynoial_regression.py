# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:54:41 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\jule\18th jule\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')

#seprating dataset into independant and depandant variable

x=dataset.iloc[:,1:2].values

y=dataset.iloc[:,2].values

#imporing linaer model to comapare with poly model

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

lin_reg.fit(x,y)
#now simple linear regression

#2.fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures(degree=6)

x_poly=pol_reg.fit_transform(x) 

pol_reg.fit(x_poly,y)

#we create 2nd object for linearregression
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)

#PLOt the observationns
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff(linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visulizaing polynomial regression results

plt.scatter(x, y,color='red')
plt.plot(x,lin_reg2.predict(pol_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff(polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


#predicting new results with linear regression
lin_reg.predict([[6.5]]) #slr

#predicting new results with polynomial regression
lin_reg2.predict(pol_reg.fit_transform([[6.5]]))

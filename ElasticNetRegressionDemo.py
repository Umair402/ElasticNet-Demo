import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  mean_absolute_error, mean_squared_error


df = pd.read_csv("AMES_Final_DF.csv")

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

elastic_model = ElasticNet()

parameters = {'alpha' : [0.001, 0.1, 0.5, 0.9, 1, 100, 1000],
             'l1_ratio' : [0.1, 0.5, 0.99, 1]}

grid_search = GridSearchCV(estimator=elastic_model,
                          param_grid=parameters,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=0)

grid_search.fit(X_train, y_train)

print(f'\nBest parameters from selected list: {grid_search.best_estimator_}')

y_pred = grid_search.predict(X_test)

print(f'\nMean absolute error: {np.round(mean_absolute_error(y_test, y_pred),2)}')

print(f'Root mean squared error: {np.round(np.sqrt(mean_squared_error(y_test, y_pred)),2)}')

print(f'The mean sale price is {np.round(df["SalePrice"].mean(),2)}\n \nThe mean squared error has a {np.round((mean_absolute_error(y_test, y_pred)/df["SalePrice"].mean()) *100,2)}% error compared to the mean')
print(f'The root mean squared error has a {np.round((np.sqrt(mean_squared_error(y_test, y_pred))/df["SalePrice"].mean()) *100,2)}% error compared to the mean')
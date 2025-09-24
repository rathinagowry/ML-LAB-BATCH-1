

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

boston=pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
print("Boston Dataset:\n")
print(boston.head())
print("Feature names:\n")
print(boston.columns)
print("Shape of Dataset:\n")
print(boston.shape)
print("Data types of features:\n")
print(boston.dtypes)
print("Summary of dataset:\n")
print(boston.describe())
print("Information:\n")
print(boston.info())

x=boston.drop('medv',axis=1)
y=boston['medv']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)

boston.hist(figsize=(12,10),bins=20,edgecolor='black')
plt.tight_layout()
plt.show()

sns.pairplot(boston[['crim','nox','dis']])
plt.show()

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(boston.corr(),annot=True,cmap='coolwarm')
plt.show()
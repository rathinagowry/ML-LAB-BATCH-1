

import numpy as np
from scipy import linalg
data=[1,2,3,4,5,6]
print("List:",data)
array_1d=np.array(data)
print("1D array using numpy",array_1d)
array_2d=array_1d.reshape(2,3)
print("Reshaped 2D array:\n",array_2d)
array_3d=array_1d.reshape(1,2,3)
print("reshaped 3D array:\n",array_3d)
random_matrix=np.random.rand(2,3)
print("Random matrix:\n",random_matrix)
matrix=np.array([[1,2],[3,4]])
detereminant=linalg.det(matrix)
print("Determinant of the matrix is",detereminant)
eigen_val,eigen_vector=linalg.eig(matrix)
print("Eigen values:\n",eigen_val)
print("Eigen vectors:\n",eigen_vector)

import pandas as pd
import matplotlib.pyplot as plt
#data=[1,2,3,4,5,6]
x=[1,3,5,7,9]
y=[2,4,6,8,10]
plt.plot(x,y,label="Line",color="blue",marker="o")
plt.title("Line")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()

data=[1,2,3,4,5,6]
index=['a','b','c','d','e','f']
series=pd.Series(data,index)
print(series)
for i in index:
  print(series[i])
print("Index:",series.index)
print("Values:",series.values)

data=[1,2,3,4,5,6]
arr=np.array(data)
series=pd.Series(arr)
print("Array using numpy",arr)
print("Series using pandas",series)

characters=['r','a','t','h','i','n','a']
index=[1,2,3,4,5,6,7]
series=pd.Series(characters,index)
print(series)
print("First letter of the name is:",series[1])

eg_data={"name":"Rathina","age":19,"course":"Information Technology"}
df=pd.DataFrame([eg_data])
print(df)

url="https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
data=pd.read_csv(url)
print(data.head(5))

import pandas as pd
import numpy as np
data1 = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, np.nan, 35, 45, 28],
    'Salary': [50000, 54000, np.nan, 62000, 58000]
}

data2 = {
    'ID': [3, 4, 5, 6],
    'Department': ['HR', 'Finance', 'IT', 'Marketing']
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
print("Original DataFrame 1:")
print(df1)
print("\nOriginal DataFrame 2:")
print(df2)
print("\nReshaped DataFrame using melt (Long Format):")
reshaped_df = pd.melt(df1, id_vars=['ID', 'Name'], value_vars=['Age', 'Salary'],
                      var_name='Attribute', value_name='Value')
print(reshaped_df)
print("\nFiltered Data (Age > 30):")
filtered_df = df1[df1['Age'] > 30]
print(filtered_df)
print("\nMerged DataFrame (on ID):")
merged_df = pd.merge(df1, df2, on='ID', how='left')
print(merged_df)
print("\nHandling Missing Values (Fill Age with mean, Salary with median):")
# Correct way to handle missing values without warning
df_filled = df1.copy()
df_filled['Age'] = df_filled['Age'].fillna(df_filled['Age'].mean())
df_filled['Salary'] = df_filled['Salary'].fillna(df_filled['Salary'].median())

print(df_filled)
print("\nMin-Max Normalization of Age and Salary:")
df_normalized = df_filled.copy()
df_normalized[['Age', 'Salary']] = (df_normalized[['Age', 'Salary']] - df_normalized[['Age', 'Salary']].min()) / (df_normalized[['Age', 'Salary']].max() - df_normalized[['Age', 'Salary']].min())

print(df_normalized)

from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head(5))
print("Mean:",iris.data.mean())
print("Median:",df.median())
print("Mode",df.mode())
print("Variance",df.var())
print("Standard Deviation",df.std())

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, scale, Binarizer


data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'first': ['Leone', 'Romola', 'Geri', 'Sandy', 'Jacenta', 'Diane-marie', 'Austen', 'Vanya', 'Giordano', 'Rozele'],
    'last': ['Debrick', 'Phinnessy', 'Prium', 'Doveston', 'Jansik', 'Medhurst', 'Pool', 'Teffrey', 'Elloy', 'Fawcett'],
    'gender': ['Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female'],
    'Marks': [50, 60, 65, 95, 31, 45, 45, 70, 36, 50],
    'selected': [True, False, False, False, True, True, True, False, False, False]
}
df=pd.DataFrame(data)
df_create=df.to_csv('data.csv',index=False)
df_read=pd.read_csv('data.csv')
print(df_read)
print("Informatio n about the dataset:",df.info())
print("Mean:",df['Marks'].mean())
print("Median:",df['Marks'].median())
print("Mode:",df['Marks'].mode())
print("Variance:",df['Marks'].var())
print("Standard Deviation:",df['Marks'].std())
print("Sum:",df['Marks'].sum())
print(df.describe())
print("value count of gender",df['gender'].value_counts())
#univariate analysis
print("Mean Marks:",df['Marks'].mean())
print("Median Marks:",df['Marks'].median())
print("Mode Marks:",df['Marks'].mode())
print("Variance Marks:",df['Marks'].var())
print("Standard Deviation Marks:",df['Marks'].std())
print("Range of marks:",df['Marks'].max()-df['Marks'].min())
#bivariate statistics
print("Marks by gender")
print("Mean",df.groupby('gender')['Marks'].mean())
print("Median",df.groupby('gender')['Marks'].median())

print("Variance",df.groupby('gender')['Marks'].var())
print("Standard Deviation",df.groupby('gender')['Marks'].std())
df2=df.copy()
print("Before label processing of the column gender:",df2['gender'])
label_encoder=LabelEncoder()
df2['gender']=label_encoder.fit_transform(df2['gender'])
print("After label processing of the column gender:",df2['gender'])
df['Marks_scaled']=scale(df['Marks'])
print("Scaled Marks:\n")
print(df[['id','Marks','Marks_scaled']])
df['Marks_bin']=Binarizer(threshold=50).fit_transform(df[['Marks']])
print(df[['id','Marks','Marks_bin']])
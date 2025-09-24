
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
iris=load_iris()
iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['target']=iris.target
print("Iris dataset:",iris_df.head())
print("Iris dataset shape:",iris_df.shape)
reshaped=np.reshape(iris.data,(150,4))
print("Reshaped data:\n",reshaped)
filtered=iris_df[iris_df['petal length (cm)']>1.5]
print("Filtered data:\n",filtered)

df1=pd.DataFrame({'Id':[1,2,3,4,5],'Name':['A','B','C','D','E']})
df2=pd.DataFrame({'Id':[1,2,3,4,5],'Age':[19,23,45,31,26]})
merged=pd.merge(df1,df2,on='Id')
print("Merged data:\n",merged)

df=pd.DataFrame({'A':[1.2,3.2,np.nan,4,np.nan,5,6]})
print("Original data:\n",df)
filled=df.fillna(0)
print("None values filled with 0:",filled)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled=scaler.fit_transform(iris_df)
print("Scaled data:\n",scaled)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_result=pca.fit_transform(iris_df.iloc[:,:-1])
pca_df=pd.DataFrame(data=pca_result,columns=['PC1','PC2'])
print("PCA result:\n",pca_df)
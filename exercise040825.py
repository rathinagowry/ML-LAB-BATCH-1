
#Binning Algorithm
def equal_width_binning(data,bins):
  min_val=min(data)
  max_val=max(data)
  range=max_val-min_val
  width=range/bins
  result=[]
  for value in data:
    bin_num=int((value-min_val)/width)
    if bin_num==bins:
      bin_num-=1
    result.append(bin_num)
  return result
sample_data=[5,7,9,10,13,15,18,21,25,30]
bin_count=5
bin_result=equal_width_binning(sample_data,bin_count)
print(bin_result)

#Min-Max Normalization
def min_max_normalization(data):
  min_val=min(data)
  max_val=max(data)
  return [(x-min_val/max_val-min_val) for x in data]
sample_data=[5,7,9,10,13,15,18,21,25,30]
normalized_data=min_max_normalization(sample_data)
print(normalized_data)

#t-Test for 2 means
def t_test(data1,data2):
  mean1=sum(data1)/len(data1)
  mean2=sum(data2)/len(data2)
  var1=sum([(x-mean1)**2 for x in data1])/(len(data1)-1)
  var2=sum([(x-mean2)**2 for x in data2])/(len(data2)-1)
  t=(mean1-mean2)/(((var1/len(data1))+(var2/len(data2)))**0.5)
  return t
sample_data1=[5,7,9,10,13,15,18,21,25,30]
sample_data2=[10,15,20,25,30,35,40,45,50,55]
t_value=t_test(sample_data1,sample_data2)
print(t_value)

#Chi-Square Test
def chi_square_test(observed,expected):
  chi=0;
  for o,e in zip(observed,expected):
    chi+=((o-e)**2)/e
  return chi
observed=[10,20,30,40]
expected=[25,25,25,25]
chi_value=chi_square_test(observed,expected)
print(chi_value)

#Confusion Matrix
def confusion_matrix(actual,predicted):
  tp=0
  tn=0
  fp=0
  fn=0
  for a,p in zip(actual,predicted):
    if a=="yes" and p=="yes":
      tp+=1
    elif a=="no" and p=="yes":
      fp+=1
    elif a=="no" and p=="no":
      tn+=1
    elif a=="yes" and p=="no":
      fn+=1
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*(precision*recall)/(precision+recall)
  return{"True Positive":tp,"True Negative":tn,"False Positive":fp,"False Negative":fn,"Accuracy":accuracy,"Precision":precision}
actual = ['yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'no']
predicted = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no']

result=confusion_matrix(actual,predicted)
print(result)

#Principal Component Ananlysis
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
Y=iris.target
X_mean=np.mean(X,axis=0)  #find mean for all columns
X_std=np.std(X,axis=0)    #find Std for all columns'
X_scaled=(X-X_mean)/X_std #standarize the data:mean=0 std=1

cov_matrix=np.cov(X_scaled.T) #construct covariance matrix by taking transpose as features/variable need to be present in rows
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix) #find eigen values and vectors
sorted=np.argsort(eigenvalues)[::-1]
eigenvalues=eigenvalues[sorted]
eigenvectors=eigenvectors[:,sorted]
PCs=eigenvectors[:,:2]
X_pca=X_scaled
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 

data = load_breast_cancer()
#print(data) # This will return a python dictonary 
# split Data 
X = data.data 
y = data.target

'''
Create a tuple of dataset 
'''
X,y = load_breast_cancer(return_X_y=True)
#print(X)
#print(y)

'''
Convert into the pandas dataframe format
'''

df = load_breast_cancer(as_frame=True).frame 
#print(df) # ==> This will return the pandas DataFrame Format 
#df.hist()
#plt.tight_layout()

'''
Skearn also capable of creating own dataset 
For Example to create a 5000 dataset with 2 features
'''
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=5000,centers=5) #centers means clusters
print(X)
print(y)
plt.scatter(X[:, 0], X[:, 1],c=y)

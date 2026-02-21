from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#X,y = load_iris(return_X_y=True)
#print(X)
#print(y)
data = load_iris()
X,y = data.data , data.target 
# train data 
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2) # 20% testing 80% training 
print(len(X_train))
print(len(X_test))

import numpy as np 
import matplotlib.pyplot as plt 

counts = np.bincount(y_train)
positions = np.arange(3)
plt.bar(positions,counts)

plt.xticks(positions,data.target_names)
plt.show()


from sklearn.model_selection import StratifiedShuffleSplit # This ensure similar distribution of different classes in the training set 

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_idx , test_idx in  split.split(X,y):
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

counts = np.bincount(y_train)
positions = np.arange(3)
plt.bar(positions,counts)
plt.xticks(positions,data.target_names)
plt.show()

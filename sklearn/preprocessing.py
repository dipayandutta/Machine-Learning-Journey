from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = load_iris(return_X_y=True)
#print(X)

# Train the Data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# Scale the data 
scaler = StandardScaler()

# fit Transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print(X_train_scaled)
'''
Extact Same using numpy 
'''

import numpy as np 
print(X_train - np.mean(X_train,axis=0)/np.std(X_train,axis=0))

'''
Min Max Scaling
'''
from sklearn.preprocessing import MinMaxScaler 

scalar = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)

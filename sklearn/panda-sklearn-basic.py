import pandas as pd 

df = pd.read_csv("students.csv")
print(df.head())

# Seprate Features(X) and Labels(y)
X = df[['hours_studied']]
y = df['passed']

# Encode the Label --> ML need Numbers not string 

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(list(zip(y,y_encoded)))

# using logistic Regression to train the model
from sklearn.linear_model import LogisticRegression 

model = LogisticRegression()
model.fit(X,y_encoded)

# Make the Prediction 
prediction = model.predict([[3]]) # If Hour is 3
probale_prediction = model.predict_proba([[3]])
print("Probability is : ",probale_prediction)
print("Encoded Prediction: ",prediction)
print("Final Prediction: ",label_encoder.inverse_transform(prediction)[0])

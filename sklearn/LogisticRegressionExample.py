import pandas as pd 

df = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6],
    "passed": ["no", "no", "no", "yes", "yes", "yes"]
})
print(df)

X = df[["hours_studied"]] # input 
y = df["passed"] # output 

# Label Encoder 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(list(zip(y,y_encoded)))

# Create The LogisticRegression of this Model 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X,y_encoded)

# Make a prediction 
prediction = model.predict([[5]])
print("Encoded  : ", prediction)

print("Final: ",label_encoder.inverse_transform(prediction)[0])

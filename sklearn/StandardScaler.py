import pandas as pd 

df = pd.read_csv("people.csv")
print(df)

# Separate features and label 
X = df[["age","salary"]] # two features
y = df["approved"]

# Encode the Target
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X,y_encoded)
print("Prediction (no scaling):",
      label_encoder.inverse_transform(model.predict([[28, 45000]]))[0])

# Adding StandardScaler 
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

model_scaled = LogisticRegression()
model_scaled.fit(X_scaled,y_encoded)

# Predict New Data 
new_person = [[28,45000]]

new_person_scaled = scalar.transform(new_person)
prediction = model_scaled.predict(new_person_scaled)

print("Predict (with Scaling): ",label_encoder.inverse_transform(prediction)[0])

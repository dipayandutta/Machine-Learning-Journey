import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("incidents.csv")
print(df)

# Features and label
X = df[["disk_used", "mem_used", "restart_count", "avc_denied"]]
y = df["root_cause"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(list(zip(y, y_encoded)))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.5, random_state=42,stratify=y_encoded
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Predict a new incident
incident = [[93, 38, 0, 1]]
prediction = model.predict(incident)

print("Predicted RCA:", encoder.inverse_transform(prediction)[0])

import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

columns = [
        "buying","maint","doors","persons",
        "lug_boot","safety","class"
        ]

df = pd.read_csv("car.data",header=None,names=columns)
#print(df.head())

# Separete feature and target 
X = df.drop("class",axis=1)
y = df["class"]

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train,X_test,y_train,y_test = train_test_split(X_encoded,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred,target_names=label_encoder.classes_))

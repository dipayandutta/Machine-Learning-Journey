from sklearn.preprocessing import LabelEncoder

y = ["disk_full","memory_pressure","disk_full","service_crash"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(y_encoded)
print(label_encoder.classes_)

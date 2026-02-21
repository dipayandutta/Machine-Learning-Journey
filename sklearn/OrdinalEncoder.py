from sklearn.preprocessing import OrdinalEncoder

X = [
        ["low"],
        ["medium"],
        ["high"]
    ]

ordinal_encoder = OrdinalEncoder()
X_encoded = ordinal_encoder.fit_transform(X)

print(X_encoded)

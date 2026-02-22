from sklearn.linear_model import LinearRegression

# Training Data
X = [[1], [2], [3], [4], [5]]   # Years of experience
y = [30000, 35000, 40000, 45000, 50000]  # Salary


# create and train model
model = LinearRegression()
model.fit(X,y)

# predict 

prediction = model.predict([[6]])
print(prediction)

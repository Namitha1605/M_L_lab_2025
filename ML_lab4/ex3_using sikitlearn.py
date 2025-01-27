import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the CSV file (assuming the data is in "data_ML.csv")
data = pd.read_csv('data_ML.csv')
print(data.head())

# Step 2: Prepare feature matrix (X) and target variable (y)
X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']].values  # Convert to numpy array
y = data['disease_score_fluct'].values  # Target variable as numpy array

# Step 3: Add intercept (bias) term to X (this is done automatically in LinearRegression)
#X_b = np.c_[np.ones((X.shape[0], 1)), X]  # We don't need to manually add the intercept with scikit-learn

# Step 4: Initialize and train the model using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(X, y)  # Fit the model to the data

# Step 5: Get the coefficients (theta) and intercept (bias term)
theta = model.coef_
intercept = model.intercept_
print("Theta values (coefficients):", theta)
print("Intercept (bias term):", intercept)

# Step 6: Make predictions using the trained model
y_pred = model.predict(X)
print("Predicted values:", y_pred[:10])  # Display first 10 predicted values

mse = mean_squared_error(y, y_pred)  # Mean Squared Error
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label="Actual vs Predicted")
sns.regplot(x=y, y=y_pred, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
plt.xlabel("Actual Disease Score Fluctuation")
plt.ylabel("Predicted Disease Score Fluctuation")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


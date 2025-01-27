import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Admission_Predict.csv')
print(data.head())

X = data[['GRE Score', 'TOEFL Score', 'University Rating']].values
y = data['CGPA'].values
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add column of ones for intercept
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta values (coefficients including intercept):", theta)
#
# Step 5: Make predictions
y_pred = X_b.dot(theta)
print("Predicted values:", y_pred[:5])
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))  # R² Score
print(mse)
print(f"R² Score: {r2}")

# Step 6: Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label="Actual vs Predicted")
sns.regplot(x=y, y=y_pred, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
plt.xlabel("Actual Disease Score Fluctuation")
plt.ylabel("Predicted Disease Score Fluctuation")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


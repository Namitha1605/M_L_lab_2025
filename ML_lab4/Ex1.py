import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file (assuming the data is in "data.csv")
data = pd.read_csv('data_ML.csv')
print(data.head())

X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']] # Features
y = data['disease_score']

X = np.c_[np.ones(X.shape[0]), X]
print(f"x value: {X}")
alpha = 0.0000001  # increase in alpha the convergence is coming
iterations = 1000
theta = np.random.randn(X.shape[1])


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_data = []
    theta_data =[]
    for i in range(iterations):
        predictions = np.dot(X, theta)
        loss = predictions - y
        cost = (1 / (2)) * np.sum(loss ** 2)
        cost_data.append(cost)
        gradients = np.dot(X.T, loss)
        theta = theta - alpha * gradients
        theta_data.append(theta)

        if i % 1 == 0:
            print(f"Iteration {i}, Cost: {cost}")

    return theta, cost_data, theta_data

def main():
    theta_final,cost_data,theta_data = gradient_descent(X, y, theta, alpha, iterations)
    print("\nFinal Parameters (Theta):", theta_final)
    predictions = np.dot(X, theta_final)
    print("\nPredictions:", predictions)
    theta_O=np.array(theta_data)
    print(f"theta: {theta_O}")
    column = theta_O[:, 0]
    #print("Selected column:", column)
    plt.scatter(column,cost_data)
    plt.show()
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")







if __name__ == "__main__":
    main()


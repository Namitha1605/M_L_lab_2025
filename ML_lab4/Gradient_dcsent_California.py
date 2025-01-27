import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Fetch the California housing dataset
california_housing = fetch_california_housing(as_frame=True)

# Check the structure of the data
print(california_housing.data.head())
print(california_housing.data.columns)
X = california_housing.data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']] # Features
y = california_housing.data['Latitude']
#
X = np.c_[np.ones(X.shape[0]), X]
print(f"x value: {X}")
alpha = 0.0000000000001  # increase in alpha the convergence is coming
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
    column = theta_O[:, 2]
    print("Selected column:", column)
    plt.scatter(column, cost_data, color='red', label='Scatter Plot')
    plt.plot(column, cost_data, color='blue', label='Line Plot')
    plt.xlabel('Theta[3]')  # Label for the x-axis
    plt.ylabel('Cost')  # Label for the y-axis
    plt.title('Theta[3] vs Cost')  # Title of the plot
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better visualization
    plt.show()






if __name__ == "__main__":
    main()


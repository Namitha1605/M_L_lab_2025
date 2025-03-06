import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EDA
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(data.columns)

X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']].values # to keep the values in data frame 
print(X)
Y = data[['disease_score']].values
print(Y)
X = np.c_[np.ones(X.shape[0]),X]
theta = np.zeros(X.shape[1])
# compute theta
def compute_cost(theta,X,Y):
    h_theta = X @ theta
    print(h_theta)
    error = h_theta - Y.flatten()
    cost = np.sum(error**2) / 2
    return cost
def compute_gradient(theta,X,Y,alpha=0.000000001,iteration=1000):
    cost_history = []
    for i in range(iteration):
        h_theta = X @ theta
        error2= h_theta-Y.flatten() # flatten is used bcz as Y is 1D and h_theta is in series transpose helps in multiplying
        theta -= alpha * ( X.T @ error2) # transpose is used as we use transpose x

        cost = compute_cost(theta,X,Y)
        cost_history.append(cost)
        if i % 1 ==0:
            print(f"the iteration:{i} and cost:{cost:.4f}")

    return theta,cost_history
theta_final,cost_hist = compute_gradient(theta,X,Y,alpha=0.000000001,iteration=1000)
# plot that represent the convergence
plt.plot(range(len(cost_hist)),cost_hist)
plt.xlabel("iteration")
plt.ylabel("cost_hist")
plt.show()

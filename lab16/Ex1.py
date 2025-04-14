# create a dataset and
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC


def load_data():
    data = np.array([[1, 13, 0],[1, 18, 0], [2, 9, 0],[3, 6, 0],[6, 3, 0],[9, 2, 0],[13, 1, 0],
        [18, 1, 0],[3, 15, 1],[6, 6, 1],[6, 11, 1],[9, 5, 1],[10, 10, 1],[11, 5, 1],[12, 6, 1], [16, 3, 1]
    ])
    X = data[:,:2]
    y = data[:,2]
    print(y)
    #plot the data points
    plt.figure(figsize=(8,6))
    plt.scatter(X[y==0,0],X[y==0,1], color='blue',label='Red Class')
    plt.scatter(X[y==0,0],X[y==1,1],color='red',label='Blue class')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()

    return X,y

# now apply the transform function and plot the graph
def transform_function(X,y):
    x1 = X[:, 0:1]  # shape (n,1)
    x2 = X[:, 1:2]
    # converts 2D to 3D
    phi = np.hstack([x1**2,x2**2,np.sqrt(2)*x1*x2])
    print(phi.T)

    return phi

def plot_3D(phi, y, clf):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(phi[y == 0, 0], phi[y == 0, 1], phi[y == 0, 2], color='blue', label='Class 0')
    ax.scatter(phi[y == 1, 0], phi[y == 1, 1], phi[y == 1, 2], color='red', label='Class 1')

    # Extract SVM weights and bias
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Create grid for decision plane
    x_range = np.linspace(phi[:, 0].min(), phi[:, 0].max(), 30)
    y_range = np.linspace(phi[:, 1].min(), phi[:, 1].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)

    # Solve for z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    # Plot decision plane
    ax.plot_surface(xx, yy, zz, alpha=0.4, color='gray', edgecolor='none')

    ax.set_xlabel('x1²')
    ax.set_ylabel('x2²')
    ax.set_zlabel('√2·x1·x2')
    ax.set_title('Decision Plane (LinearSVC) in 3D Feature Space')
    ax.legend()
    plt.show()

X,y = load_data()
phi = transform_function(X,y)


# Optional: Scale features (good practice for SVM)
scaler = StandardScaler()
phi_scaled = scaler.fit_transform(phi)

clf = LinearSVC()
clf.fit(phi_scaled, y)

plot_3D(phi_scaled, y, clf)






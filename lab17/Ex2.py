import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt



def load_data():
    x ,y = load_iris(return_X_y=True, as_frame=True)
    print(x.shape,y.shape)
    print(y)
    df = pd.DataFrame(x,y)
    print(df)
    print(df.isnull().sum())
    mask = (y==0)|(y==1)
    x_filtered = x[mask]
    y_filtered = y[mask]
    x_filtered = x_filtered.iloc[:, :2]
    print(x_filtered,y_filtered)

    return x_filtered,y_filtered

def split_train_test(x_filtered,y_filtered):

    x_train,x_test,y_train,y_test = train_test_split(x_filtered,y_filtered, test_size=0.3)

    return  x_train,x_test,y_train,y_test


def svm(x_train,x_test,y_train,y_test):
    svm = SVC(kernel= 'rbf',degree=2, random_state=50)
    svm.fit(x_train,y_train)
    ypred = svm.predict(x_test)
    accuracy_sco = accuracy_score(y_test,ypred)
    print(accuracy_sco)

    return svm


def plot_decision_boundary(svm, X, y):
    # Create a mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Stack for prediction
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(mesh_data)
    Z = Z.reshape(xx.shape)

    # Filled contour (regions)
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='coolwarm')

    # Actual decision boundary and margins
    contour = plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    plt.clabel(contour, inline=True)

    # Plot original points
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    # Mark support vectors
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

    # Label and show
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary with RBF Kernel')
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Run the full pipeline ===
x_filtered, y_filtered = load_data()
x_train, x_test, y_train, y_test = split_train_test(x_filtered, y_filtered)
model = svm(x_train, x_test, y_train, y_test)
plot_decision_boundary(model, x_train, y_train)



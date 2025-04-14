#Consider the following dataset. Implement the RBF kernel.
#   Check if RBF kernel separates the data well and compare it with the Polynomial Kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Step 1: Data
X = np.array([
    [6, 5],
    [6, 9],
    [8, 6],
    [8, 8],
    [8, 10],
    [9, 2],
    [9, 5],
    [10, 10],
    [10, 13],
    [11, 5],
    [11, 8],
    [12, 6],
    [12, 11],
    [13, 4],
    [14, 8]
])

y_labels = [
    "Blue", "Blue", "Red", "Red", "Red",
    "Blue", "Red", "Red", "Blue", "Red",
    "Red", "Red", "Blue", "Blue", "Blue"
]

# Step 2: Encode labels (Blue=0, Red=1)
le = LabelEncoder()
y = le.fit_transform(y_labels)

# Step 3: Define and train SVMs
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_poly = SVC(kernel='poly', degree=2, gamma='scale')  # degree can be adjusted

svm_rbf.fit(X, y)
svm_poly.fit(X, y)

# Step 4: Plot decision boundaries
def plot_svm_decision_boundary(model, title):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_svm_decision_boundary(svm_rbf, "SVM with RBF Kernel")

plt.subplot(1, 2, 2)
plot_svm_decision_boundary(svm_poly, "SVM with Polynomial Kernel (deg=3)")

plt.tight_layout()
plt.show()

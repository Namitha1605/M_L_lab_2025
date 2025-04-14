import numpy as np

# Step 1: Define the kernel function
def polynomial_kernel(a, b):
    return a[0]**2 * b[0]**2 + 2 * a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2

# Step 2: Define the transformation phi
def phi(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])

# Step 3: Choose two sample points
x1 = np.array([1, 2])
x2 = np.array([3, 4])

# Step 4: Apply kernel trick
K_output = polynomial_kernel(x1, x2)

# Step 5:  transform and compute dot product
phi_x1 = phi(x1)
phi_x2 = phi(x2)
dot_product = np.dot(phi_x1, phi_x2)

# Step 6: results
print("Kernel Output (without transforming):", K_output)
print("Dot Product after transformation φ(x):", dot_product)

import numpy as np

def phi(x):
    x1, x2 = x
    return np.array([x1**2, x2**2, np.sqrt(2) * x1 * x2])

def main():
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    # With explicit transformation
    x1_sq = phi(x1)
    x2_sq = phi(x2)
    dot_prod_transformed = np.dot(x1_sq, x2_sq)

    # Without transformation (using kernel trick)
    dot_prod_kernel = np.dot(x1, x2)**2
 
    print(f"Transformed x1: {x1_sq}")
    print(f"Transformed x2: {x2_sq}")
    print(f"Dot product in transformed space: {dot_prod_transformed}")
    print(f"Dot product using kernel trick: {dot_prod_kernel}")

if __name__ == '__main__':
    main()

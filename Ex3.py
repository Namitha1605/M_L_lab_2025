import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = 2 * x ** 2 + 3 * x + 4

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
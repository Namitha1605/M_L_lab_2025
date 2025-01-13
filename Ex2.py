# y = 2x + 3 and plot x , y (start -100, stop 100, num =100)
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-100, 100, 100)
y = 2 * x + 3
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



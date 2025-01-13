import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2
def df(x):
    return 2 * x

x = np.linspace(-100,100,100)

y = f(x)
dy = df(x)

plt.plot(x, y)
plt.plot(x,dy)
plt.xlabel('x')
plt.ylabel('y')
plt.xlabel('x')
plt.ylabel('dy')

plt.show()
#A transpose
import numpy as np
from numpy.ma.core import multiply

A = np.array ([[1,2,3],[4,5,6]])
result = np.transpose(A)
print(result)
multi = result
AAT = result @ A
print(AA_T)













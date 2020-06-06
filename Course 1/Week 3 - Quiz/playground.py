import numpy as np

A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
C = np.sum(A, axis = 0, keepdims = True)

# print(A)
# print("")
# print(B)
# print("")
# print(C)
# print(B.shape)



D = np.matrix('10 20 30 ')
E = D/10
print(D)
print(E)
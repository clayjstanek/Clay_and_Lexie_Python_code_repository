import numpy as np

# Chapter 15 Solution Starter

A = np.array([[4,2],[1,3]],dtype=float)
eigvals,eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)

Q = eigvecs
L = np.diag(eigvals)
print(Q @ L @ np.linalg.inv(Q))

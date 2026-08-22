import numpy as np

# Chapter 16 Solution Starter

A = np.array([[3,0],[0,1]],dtype=float)
U,S,Vt = np.linalg.svd(A)
print(U)
print(S)
print(Vt)

Sigma = np.zeros_like(A)
Sigma[0,0]=S[0]
Sigma[1,1]=S[1]
print(U @ Sigma @ Vt)

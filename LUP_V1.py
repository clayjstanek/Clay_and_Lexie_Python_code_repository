# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:26:17 2026

@author: cstan
"""

# LU decomposition
from numpy import array
from scipy.linalg import lu
# define a square matrix
A = array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print(A)
# factorize
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)
print(B)
B = P@(L)@(U)
print(B)
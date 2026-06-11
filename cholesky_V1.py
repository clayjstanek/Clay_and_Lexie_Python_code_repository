# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:36:36 2026

@author: cstan
"""

# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define symmetrical matrix
A = array([
[2, 1, 1],
[1, 2, 1],
[1, 1, 2]])
print(A)
# factorize
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)

B = L@(L.T)
print(B)
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:48:41 2026

@author: annil
"""

import numpy as np

"""
Chapter 9 Homework: matrices
"""

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

# ========================================
# Part A - Python Skills Reinforcement
# ========================================

section('Part A - Python Skills Reinforcement')

# Create two 2x3 matrices
A = np.array([[1, 3, 5],
              [1, 3, 5]])
B = np.array([[2, 4, 6], 
              [2, 4, 6]])

# Print matrices
print('\nMatrix A:', A)
print('Matrix B:', B)

# Add matrices 1 and 2
print('\nShape of A:', A.shape)
print('Shape of B:', B.shape)
print('A + B =', A + B)

# Subtract matrices 1 and 2
print('\nShape of A:', A.shape)
print('Shape of B:', B.shape)
print('A - B =', A - B)

# Hadamard multiplication
print('\nShape of A:', A.shape)
print('Shape of B:', B.shape)
print('A * B =', A * B)

# Element-wise division
print('\nShape of A:', A.shape)
print('Shape of B:', B.shape)
print('A / B =', A / B)

# Create 3x2 and 2x2 matrices
C = np.array([[1, 2],
              [3, 4],
              [5, 6]])
D = np.array([[4, 3],
              [2, 1]])

# Print matrices
print('\nMatrix C:', C)
print('Matrix D:', D)

# Multiply C and D
print('\nShape of C:', C.shape)
print('Shape of D:', D.shape)
try:
    print('C * D =', C * D)
except Exception as e:
    print('Could not multiply C and D. Broadcating error:', e)

# Multiply A by vector [1, 2]
print('\nShape of A:', A.shape)
print('Shape of [1, 2]:', np.array([1, 2]).shape)
try:
    print('A * [1, 2] =', A * np.array([1,2]))
except Exception as e:
    print('Could not multiply A and [1, 2]. Broadcasting error:', e)

# Multiply A by scalar, 2
print('\nShape of A:', A.shape)
print('Shape of 2:', np.array([2]).shape)
print('A * 2 =', A * 2)

# ========================================
# Part B - Practical Problem
# ========================================

section('Part B - Practical Problem')

# Create scores matrix
A = np.array([[85, 90],
              [78, 88],
              [92, 95]])

# Print scores matrix
print('Scores:', A)

# Exam weights 
w = np.array([[0.4],
              [0.6]])

# Print weights
print('Exam weights:', w)

# Weighted scores using matrix-vector multiplication
scores_weighted = A.dot(w)
print('Weighted scores:', scores_weighted)

# Why marix multiplication is a natural representation of weighted averaging
"""
Matrix multiplication is a natural representation of weighted 
averaging because it multiplies each score by their weights and 
adds them together.
"""

# Scores matrix with third exam
third_exam_scores = np.array([[93],
                              [89],
                              [97]])
B = np.hstack((A, third_exam_scores))

# Print matrix with all three exam scores
print('Matrix with third exam scores:', B)

# Interpret the dimensions of every matrix and vector involved
"""
A is a 3x2 matrix, and w is a 2x1 matrix so they can be multiplied 
using a dot product, and the result is a 3x1 matrix. B is a 3x1 
matrix, so it can be stacked horizontally with A, resulting in a 
3x3 matrix.
"""

# Explain how this resembles the matrix form of linear regression discussed in class
"""
This resembles the matrix form of linear regression discussed in 
class because if each column is a 3x1 vector, then taking the dot 
product of the matrix and weight vector basically just gives us a 
vector that is a weighted average of the other vectors. 
"""


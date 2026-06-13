# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:57:29 2026

@author: annil
"""

import numpy as np
from scipy.linalg import lu
from numpy.linalg import qr, cholesky, matrix_rank, det, norm

"""
Chapter 14 Homework - Matrix Decompositions
"""

def section(title):
    print('\n' +  '=' * 38 + '\n' + title + '\n' + '=' * 38)

# =======================================
# Part A - Python Skills Reinforcement
# =======================================

def partA_python_skills_reinforcement():
    section('Part A - Python Skills Reinforcement')
    
def problem1_lu_decomposition():
    print('\nProblem 1 - LU Decomposition')
    
    # Create A
    A = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
        ])
    print('\nMatrix A:\n', A)
    
    # Decompose A
    P, L, U = lu(A)
    
    # Print matrices
    print('\nMatrix P:\n', P)
    print('\nMatrix L:\n', L)
    print('\nMatrix U:\n', U)
    
    # Reconstruct A
    print('\nReconstructed A with dot product of P, L, U:\n', P @ L @ U)
    
def problem2_qr_decomposition():
    print('\nProblem 2 - QR Decomposition')
    
    # Create A
    A = np.array([
        [1, 2], 
        [3, 4],
        [5, 6]
        ])
    print('\nMatrix A:\n', A)
    
    # Decompose A
    Q, R = qr(A)
    
    # Print matrices
    print('\nMatrix Q:\n', Q)
    print('\nMatrix R:\n', R)
    
    # Reconstruct A
    print('\nReconstructed A with dot product of Q and R:\n', Q @ R)
    
    # Reconstruction error
    print('\nReconstruction error:', norm(A - (Q @ R)))
    
def problem3_cholesky_decomposition():
    print('\nProblem 3 - Cholesky Decomposition')
    
    # Create A
    A = np.array([
        [4, 2, 2],
        [2, 5, 1],
        [2, 1, 3]
        ])
    print('\nA:\n:', A)
    
    # Cholesky Decomposition
    L = cholesky(A)
    print('\nL:\n', L)
    
    # Reconstruct A using L @ L.T
    print('\nReconstructed A using L @ L.T:\n', L @ L.T)
    
    # Explain why A must be symmetric
    print('\nA must be symmetric because in order to use '
          'cholesky decomposition, it has to be a square '
          'matrix with positive values that is symmetric. '
          'The reconstruction L @ L.T will always be '
          'symmetric, so it has to be symmetric to begin '
          'with.')
    
def problem4_compare_shapes():
    print('\nProblem 4 - Compare Shapes')
    
    # LU decomposition
    A = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
        ])
    P, L, U = lu(A)
    print('\nP.shape:', P.shape)
    print('L.shape:', L.shape)
    print('U.shape:', U.shape)
    
    # QR decomposition
    A = np.array([
        [1, 2], 
        [3, 4],
        [5, 6]
        ])
    Q, R = qr(A)
    print('\nQ.shape:', Q.shape)
    print('R.shape:', R.shape)
    
    # Cholesky decomposition
    A = np.array([
        [4, 2, 2],
        [2, 5, 1],
        [2, 1, 3]
        ])
    L = cholesky(A)
    print('\nL.shape:', L.shape)
    
    # Explain why the shapes differ and when each type is applicable
    print('\nLU decomposition is used for square matrices '
          'and it produces square matrices as factors which '
          'match the shape of A. QR decomposition is used '
          'for rectangular matrices. Q has the same number of '
          'rows as A or less depending on if it is reduced '
          'or not, and R is an upper triangle matrix with '
          'the same number of columns as A. Cholesky '
          'decomposition can only be used for symmetric '
          'square matrices with positive values and it '
          'produces one triangular matrix L and implicity, '
          'L.T.')
    
def partB_practical_problem_sensor_correlation_compression():
    section('Part B - Practical Problem: Sensor Correlation Compression')
    
    A = np.array([
        [12, 15, 18],
        [14, 17, 22],
        [16, 20, 25]
        ])
    
    # Rank of A
    print('\nRank of A:', matrix_rank(A))
    
    # Determinant of A
    print('\nDeterminant of A:', det(A))
    
    # Interpretation
    print('\nThe matrix has a rank of 3 which means that it '
          'has 3 linearly independent columns and spands 3 '
          'dimensions. The determinant is not close to zero '
          'so the columns are not exact linear combinations '
          'of each other.')
    
    # QR decomposition
    Q, R = qr(A)
    print('\nQ:\n', Q)
    print('\nR:\n', R)
    
    # What information is contained in Q and R?
    print('\nQ basically just tells us which directions the '
          'columns of A go in with normal vectors that are '
          'orthogonal. R tells how to combine those vectors '
          'to get A.')
    
    # Reconstruct A
    print('\nA Reconstructed:\n', Q @ R)
    
    # Reconstruction error
    print('\nReconstruction error:', A - (Q @ R))
    
    # Explain why factorization methods are preferable to directly computing inverses
    print('\nFactorization methods are preferable to '
          'directly computing inverses for larges matrices '
          'becasue it can be slow and hard to do. '
          'Decomposing the matrices breaks them into pieces '
          'that are easier to work with.')
    
    # Research Question
    print('\nMatrix decomposition is considered one of the '
          'most important tools in numerical linear algebra '
          'because it improves computational efficiency of '
          'linear algebra problems such as least-squares '
          'problems. It also helps numerical stability by '
          'reducing errors of computations. Decompositions '
          'appear throughout machine learning because they '
          'make large-scale linear algebra much more '
          'computationally feasible')

def main():
    partA_python_skills_reinforcement()
    problem1_lu_decomposition()
    problem2_qr_decomposition()
    problem3_cholesky_decomposition()
    problem4_compare_shapes()
    partB_practical_problem_sensor_correlation_compression()
    
if __name__ == '__main__':
    main()
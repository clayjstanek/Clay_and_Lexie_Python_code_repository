# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:10:11 2026

@author: annil
"""
import numpy as np

"""
Chapter 15 Homework - Eigenvalues, Eigenvectors, and Spectral Theory
"""

def section(title):
    print('\n' + '=' * 40 + '\n' + title + '\n' + '=' * 40)

def partA_computational_practice():
    section('Part A - Computational Practice')
    
    # Create matrices
    A1 = np.array([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2]
        ])
    A2 = np.array([
        [3, -2, 1],
        [-2, 4, -1],
        [1, -1, 2]
        ])
    A3 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
        ])
    A4 = np.array([
        [6, 2, 1],
        [2, 5, 2],
        [1, 2, 4]
        ])
    A5 = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 2, 3]
        ])
    
    matrices = [A1, A2, A3, A4, A5]
    
    # Show matrices
    i = 1
    print('\n')
    for matrix in matrices:
        print('A' + str(i) + ' :\n' , matrix)
        i += 1
    
    # Compute ranks
    i = 1
    print('\n')
    for matrix in matrices:
        print('A' + str(i) + ' rank:' , np.linalg.matrix_rank(matrix))
        i += 1
    
    # Compute determinants
    i = 1
    print('\n')
    for matrix in matrices:
        print('A' + str(i) + ' determinant:' , np.linalg.det(matrix))
        i += 1
        
    # Verify determinant = product(eigenvalues)
    i = 1
    print('\n')
    for matrix in matrices:
        values, vectors = np.linalg.eig(matrix)
        print('A' + str(i) + ' determinant with eigenvalues product:' , np.prod(values))
        i += 1
    
    # Compute traces
    i = 1
    print('\n')
    for matrix in matrices:
        print('A' + str(i) + ' trace:' , np.linalg.trace(matrix))
        i += 1
        
    # Verify trace = sum(eigenvalues)
    i = 1
    print('\n')
    for matrix in matrices:
        values, vectors = np.linalg.eig(matrix)
        print('A' + str(i) + ' trace with eigenvalues sum:' , sum(values))
        i += 1
    
    # Compute eigenvalues and eigenvectors
    i = 1
    for matrix in matrices:
        values, vectors = np.linalg.eig(matrix)
        print('\nA' + str(i) + ' eigenvalues:\n' , values)
        print('A'+ str(i) + ' eigenvectors:\n' , vectors)
        i += 1
    
    # Verify A = Q * Lambda * Q^-1
    i = 1
    print('\n')
    for matrix in matrices:
        values, vectors = np.linalg.eig(matrix)
        print('A' + str(i) + ' reconstructed:\n' , vectors.dot(np.diag(values)).dot(np.linalg.inv(vectors))) 
        i += 1
       
def partC_3x3_spectral_decomposition():
    section('Part C - 3x3 Spectral Decomposition')
    
    # Create matirx
    A = np.array([
        [3, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
        ])
    
    print('\nA:\n', A)
    
    # Eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(A)
    print('\nEigenvalues:\n', values)
    print('\nEigenvectors:\n', vectors)
    
    # Q, Lambda, Q^-1, and Q^T
    Q = vectors
    print('\nQ:\n', Q)
    
    Lambda = np.diag(values)
    print('\nLambda:\n', Lambda)
    
    Q_inverse = np.linalg.inv(Q)
    print('\nQ^-1:\n', Q_inverse)
    
    Q_transpose = Q.T
    print('\nQ^T:\n', Q_transpose)
    
    # Explanation
    print('\nThe inverse and transpose of Q are the same '
          'because Q is just the identity matrix. Therefore, '
          'the matrix that undoes Q, and the matrix that is '
          'made by switching the rows and columns are the '
          'exact same.')
    
def partD_spectral_application():
    section('Part D - Spectral Applications')
    
    # Create matirx
    A = np.diag([2, 3, 5])
    print('\nA:\n', A)
    
    # Eigen decomposition
    values, vectors = np.linalg.eig(A)
    Q = vectors
    Q_inv = np.linalg.inv(Q)
    Lambda = np.diag(values)
    
    # Compute A^3, A^10, and A^50
    print('\nA^3:\n', Q @ (Lambda ** 3) @ Q_inv)
    print('\nA^10:\n', Q @ (Lambda ** 10) @ Q_inv)
    print('\nA^50:\n', Q @ (Lambda ** 50) @ Q_inv)
    
    # Fibonacci matrix
    B = np.array([
        [1, 1],
        [1, 0]
        ])
    
    values, vectors = np.linalg.eig(B)
    Q = vectors
    Q_inv = np.linalg.inv(Q)
    Lambda = np.diag(values)
    
    """
    I'm not really sure what I'm supposed to do with this 
    decomposition, so I'm just writing some observations: 
    I'm guessing that the matrix decomposition makes it 
    easier to raise this matrix to a power and that it might 
    help us to see why the pattern emerges.
    """
    
    # Verify trace/determinant theorems using the diagonal matrix of [2, 3, 5] from earlier
    A = np.diag([2, 3, 5])
    values, vectors = np.linalg.eig(A)
    
    trace = np.sum(values)
    print('\nTrace of A:', trace)
    print('Does our trace match the numpy trace?', np.isclose(trace, np.linalg.trace(A)))
    
    determinant = np.prod(values)
    print('\nDeterminant of A:', determinant)
    print('Does our determinant match the numpy determinant?', np.isclose(determinant, np.linalg.det(A)))
    
def partE_power_method():
    section('Part E - Power Method')
    
    """
    We didn't really go over this, but from what I could 
    gather from other sources, the power method is just 
    repeatedly multiplying and normalizing matrices by a in 
    order to get it close to the dominant eigenvector.
    """
    
    A = np.array([
        [5, 1],
        [1, 2]
        ])
    
    x = np.array([1, 1])
    x_0 = x.T
    
    x_1 = A @ x_0
    x_1 = x_1 / np.max(np.abs(x_1))
    
    x_2 = A @ x_1
    x_2 = x_2 / np.max(np.abs(x_2))
    
    x_3 = A @ x_2
    x_3 = x_3 / np.max(np.abs(x_3))
    
    x_4 = A @ x_3
    x_4 = x_4 / np.max(np.abs(x_4))
    
    x_5 = A @ x_4
    x_5 = x_5 / np.max(np.abs(x_5))
    
    print('\nPower method approx dominant eigenvector:', x_5)
    
    values, vectors = np.linalg.eig(A)
    
    idx = np.argmax(values)
    v_dom = vectors[:, idx]
    
    print('\nActual dominant eigenvector:', v_dom)
    
def partF_markov_chain():
    section('Part F - Markov Chain')
    
    P = np.array([
        [0.9, 0.2],
        [0.1, 0.8]
        ])
    
    # Find steady-state distribution 
    values, vectors = np.linalg.eig(P)
    
    idx = np.argmin(np.abs(values - 1))
    steady = vectors[:, idx]
    
    # Normalize
    steady = steady / np.sum(steady)
    print('\nSteady-state distribution:', steady)
    
    """
    This steady-state distribution means that over time, the 
    system stabilizes with 2/3 in state 1 and 1/3 in state 2.
    """
    
def main():
    partA_computational_practice()
    partC_3x3_spectral_decomposition()
    partD_spectral_application()
    partE_power_method()
    partF_markov_chain()
    
if __name__ == '__main__':
    main()
    
    
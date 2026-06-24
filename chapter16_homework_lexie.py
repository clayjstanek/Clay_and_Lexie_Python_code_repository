# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 08:07:16 2026

@author: annil
"""

import numpy as np
from sklearn.datasets import load_iris

"""
Chapter 16 Homework - Singular Value Decomposition
"""

def section(title):
    print('\n' + '=' * 40 + '\n' + title + '\n' + '=' * 40)
    
def partA_computational_svd():
    section('Part A - Computational SVD')
    
    # SVD for the following matrices
    S1 = np.array([
        [3, 1], 
        [1, 3]
        ])
    S2 = np.array([
        [1, 2], 
        [3, 4],
        [5, 6]
        ])
    S3 = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
        ])
    S4 = np.array([
        [1, 2],
        [2, 4],
        [3, 6]
        ])
    S5 = np.array([
        [10, 10, 10],
        [10, 10, 10],
        [1, 1, 1]
        ])
        
    matrices = [S1, S2, S3, S4, S5]
    
    # SVD Decomposition for each matrix
    i = 1
    for matrix in matrices:
        # Print matrix
        print('\nS' + str(i) + ':\n', matrix)
        
        # Decomposition
        U, sigma, VT = np.linalg.svd(matrix)
        print('\nU:\n', U)
        print('\nsigma:\n', sigma)
        print('\nVT:\n', VT)
        
        # Reconstruct
        sigma_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
        sigma_matrix[:matrix.shape[1], :matrix.shape[1]] = np.diag(sigma)
        print('\nReconstructed matrix:\n', U @ sigma_matrix @ VT)
        
        
        # Rank
        print('\nRank:', np.linalg.matrix_rank(matrix))
        
        # Interpret singular values
        """
        Sigma is an mxn diagonal matrix. The np output is a 
        1D array and the values of the array are scalars for 
        how much the matrix stretches in each direction. The 
        number of nonzero singular values for sigma is the 
        rank of the matrix
        """
        rank = np.sum(np.sum(~np.isclose(0, sigma)) )
        print('Rank from sigma:', rank)
        
        i += 1
        
def partB_check_hand_SVD():
    section('Part B - Check Hand SVD')
    
    A = np.array([
        [3, 0],
        [0, 1]
        ])
        
    # Compute A^TA
    ATA = A @ A.T
    print('\nA^TA:\n', ATA)
    
    # Eigenvalues
    values, vectors = np.linalg.eig(ATA)
    print('\nEigenvalues:', values)
    
    # Singular values
    U, sigma, VT = np.linalg.svd(A)
    print('\nU:\n', U)
    print('Sigma:\n', sigma)
    print('VT:\n', VT)
    
    """
    Matches hand calculations! Thank goodness. They were very difficult.
    """
    
def partC_rank1_approximation():
    section('Part C - Rank-1 Approximation')
    
    S5 = np.array([
        [10, 10, 10],
        [10, 10, 10],
        [1, 1, 1]
        ])
    
    # Eigendecomposision
    U, sigma, VT = np.linalg.svd(S5)
    
    # Max singular value, first column of U, first rot of V^T
    sigma_1 = sigma[0]
    print('\nLargest singular value:\n', sigma_1)
    U_1 = U[:,0].reshape(-1,1)
    print('U_1:\n', U_1)
    VT_1 = VT[0,:].reshape(1,-1)
    print('V^T_1:\n', VT_1)
    
    # Rank-1 approximation
    print('\nRank-1 approximation:\n', sigma_1 * (U_1 @ VT_1))
    
    # Explanation
    """
    This approximation effectively did not change this matrix 
    because it was already rank 1 to begin with since all of 
    the columns and rows are just scaled versions of each 
    other. For a matrix of rank 2 or higher, however, this 
    compression would havce kept the shape of the matrix and 
    it would have kept the most dominant direction and scalar 
    which is useful becuase it takes up much less space, and 
    it is easier to do calculations with the compressed 
    version.
    """
    
def partD_pca():
    section('Part D - PCA')
    
    iris = load_iris()
    X = iris.data
    
    # Feature names
    print('\nIris feature names:', iris.feature_names)
    
    # Create 4x4 covariance matrix with centered data
    X_centered = X - X.mean(axis=0)
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    print('\nCovariance matrix:\n', cov)
    
    """
    This diagonal entries show variance of each attribute, 
    and the other entries of the matrix show how each 
    combination of 2 attributes co-vary. If the entry is 
    positive, they increase together, and if the entry is 
    negative, one increases while the other decreases.
    """
    
    # Eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)
    print('\nEigenvalues:', values)
    print('\nEigenvectors:\n', vectors)
    
    """
    The eigenvalues tell us how many units of variance in a 
    given direciton. The eigenvectors describe each direction. 
    Since the values are ranked from largest to smallest, the 
    vectors also go in order of most to least informative. 
    Each vector has a value for each feature, so for example, 
    the most dominant direction is the vector 
    v_1 = [0.36, -0.08, 0.86, 0.35]. The largest value 
    corresponds to petal length, so species differ most in 
    petal length.
    """
    
    # SVD
    U, s, VT = np.linalg.svd(X_centered, full_matrices=False)
    print('\n(s ** 2) / (n - 1)\n:', (s ** 2) / (X_centered.shape[0] - 1))
    print('\nV:', VT.T)
    
    """
    Using SVD gives us a different way of finding the 
    eigenvalues and eigenvectors. Since sigma = sqrt(lambda) 
    when they are normalized, we can just square sigma and 
    divided by (n-1) to get eigenvalues, and the eigenvectors 
    are just V which we find by taking the transpose of V^T.
    """
    
    # Interpretation of principal components
    """
    PC1:
        The first principal component is dominated by petal 
        length and width. Sepal length has moderate weight in 
        this principal component. PC1 explains the largest 
        amount of variance between species samples, and describes 
        overall petal size. 
    PC2:
        The strongest weights for PC2 are sepal width and sepal 
        length, and their sign is different from petal measurements, 
        so sepal measurements contrast against petal size. PC2 
        shows variation in sepal shake that is independent of the 
        petal-size variation in PC1.
        
    PC3 and PC4:
        These components have very small eigenvalues compared to 
        PC1 and PC2. This means that they have very little variance 
        and aren't really meaningful.'
    """
    
def partE_netflix_problem():
    section('Part E - Netflix Problem')
        
    # Rating array with missing placeholder 0
    R = np.array([
        [5, 4, 0],
        [4, 0, 2],
        [0, 1, 4],
        [3, 0, 0]
        ])
    print('\nR:\n', R)
    
    # SVD
    U, s, VT = np.linalg.svd(R, full_matrices = False)
    S = np.zeros((R.shape[0], R.shape[1]))
    S[:R.shape[1], :R.shape[1]] = np.diag(s)
    
    # Rank 2 approximation
    k = 2
    U_k = U[:, :k]
    S_k = S[:k, :k]
    VT_k = VT[:k, :]
    
    R_approx = U_k @ S_k @ VT_k
    print('\nRank 2 approximation of R:\n', R_approx)
    
    """
    The latent factors are basically just patterns in how user 
    preferences and movie characteristics are related. Low-rank 
    approximation estimates missing values by using known ratings 
    to fill in unknown ratings based on the latent factors.
    """
    
def main():
    partA_computational_svd()
    partB_check_hand_SVD()
    partC_rank1_approximation()
    partD_pca()
    partE_netflix_problem()
        
if __name__ == '__main__':
    main()
        
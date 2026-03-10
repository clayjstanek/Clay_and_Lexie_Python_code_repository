#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:10:55 2026

@author: lexienilles
"""

import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


# ------------------------------
# Part A - Core Exercises
# ------------------------------

def print_vector_norm(list):
    for vector in list:
        print("\nEuclidean norm of", vector , ":")
        print(np.linalg.norm(vector))


def triangle_inequality(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a+b) <= np.linalg.norm(a) + np.linalg.norm(b)
    

def angle_between_vectors(a: np.ndarray, b: np.ndarray):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos_theta = np.dot(a, b) / (na * nb)
    theta = math.acos(cos_theta)
    return theta * 180 / math.pi


def project(u: np.ndarray, v: np.ndarray):
    nv = float(np.linalg.norm(v))
    if nv != 0:
        return (float(np.dot(u, v)) / nv ** 2) * v
    
    
def verify_residual_is_orthogonal(u: np.ndarray, v: np.ndarray):
    residual = u - project(u, v)
    return int(np.dot(residual, v)) == 0
    

    
# A1. Vector Norms and Geometry

vector1 = np.array([0, 6.0 ,7.0])
vector2 = np.array([3.0, 6.0, 9.0])
vector3 = np.array([1.0, 1.0, 4.0])
   
list_of_vectors = [vector1, vector2, vector3]
    
# Print Euclidean norm for each vector
print_vector_norm(list_of_vectors)
    
# Print whether triangle inequality is true for vectors 1 and 2   
print("\nTriangle inequality for vectors 1 and 2 is:")
print(triangle_inequality(vector1, vector2))
    
# Print angle between two vectors using cosine similarity
print("\nAngle between vectors 1 and 2 in degrees:")
print(angle_between_vectors(vector1, vector2))
    
# A2. Projection and Orthogonality
    
# Random vectors u and v
vector_u = np.random.randn(2)
vector_v = np.random.randn(2)
    
# Project vector_u onto vector_v and verify
print("\nVector u:", vector_u)
print("Vector v:", vector_v)
print("Vector u projected onto v:")
print(project(vector_u, vector_v))
print("Is residual orthogonal to v:")
print(verify_residual_is_orthogonal(vector_u, vector_v))
    
# A3. Matrix Operations

# Random 3x3 matrix A and vector b
matrix_A = np.random.randn(3, 3)
vector_b = np.random.randn(3)
    
# Solve Ax = b
print("\nMatrix A:")
print(matrix_A)
print("Vector b:")
print(vector_b)
print("For Ax = b,")
x = np.linalg.solve(matrix_A, vector_b)
print("x =", x)
print("Check Ax:", matrix_A @ x)
print("Determinant:", np.linalg.det(matrix_A))
        
    
# ------------------------------------------
# Part B - Mini Project: Iris Dataset
# ------------------------------------------

iris = load_iris()
iris_data = iris.data    

# Normalize vectors
norms = np.linalg.norm(iris_data)
iris_norm = iris_data / norms

# Cosine similarity between 1st and 2nd samples
cos_sim = np.dot(iris_norm[0], iris_norm[1])
print("\nCosine similarity between 1st and 2nd samples:", cos_sim)

# Plot first two features 
plt.figure(figsize=(3,1.5))
plt.scatter(iris_data[:,0], iris_data[:,1], s=6)
plt.title("First Two Features of Iris Data")
plt.xlabel("1st Feature")
plt.ylabel("2nd Feature")
plt.show()
"""
There seems to be clusters of data in the middle and on the left of the plot. The fact that there are some clusters on the scatterplot means that the first two features may be loosely correlated, but its kind of hard to tell from just this graph.
"""

# Reflection questions
"""
#1
Normalization matters in cosine similarity because comparison of the vectors is more accurate if they're both the same length, and an easy way to do that is just to make them 1 unit long. Also, when the vectors are normalized, cosine similarity is easier to calculate because its just the dot product.
#2
Orthogonality means that the data has no correlation because they're not going in the same direction at all since they're perpendicular.
#3
Matrix multiplication encodes geometric transformations because multiplying by a matrix allows you to change the components in pretty much any way, like scaling, rotating, projecting, etc. by 
"""


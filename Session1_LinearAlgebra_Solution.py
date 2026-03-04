# Session 1 Solution – Linear Algebra & Geometry

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ---------- Part A ----------

# A1: Norms and angle
v = np.array([1, 2, 3], dtype=float)
w = np.array([4, -1, 2], dtype=float)

print("||v|| =", np.linalg.norm(v))
print("||w|| =", np.linalg.norm(w))
print("Triangle inequality check:", np.linalg.norm(v+w),
      "<=", np.linalg.norm(v) + np.linalg.norm(w))

cos_sim = np.dot(v, w) / (np.linalg.norm(v)*np.linalg.norm(w))
angle = np.arccos(cos_sim)
print("Angle (radians):", angle)

# A2: Projection
def project(u, v):
    return (np.dot(u, v)/np.dot(v, v)) * v

u = np.array([3, 1])
v2 = np.array([1, 2])
proj = project(u, v2)
residual = u - proj
print("Residual dot v:", np.dot(residual, v2))

# A3: Matrix solve
A = np.random.randn(3,3)
b = np.random.randn(3)
x = np.linalg.solve(A, b)
print("Check Ax:", A @ x)
print("Determinant:", np.linalg.det(A))

# ---------- Part B (Iris example) ----------

iris = load_iris()
X = iris.data

# Normalize rows
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / norms

# Cosine similarity between first two samples
sim = np.dot(X_norm[0], X_norm[1])
print("Cosine similarity sample 0 & 1:", sim)

# Plot first two dimensions
plt.scatter(X[:,0], X[:,1], c=iris.target)
plt.title("Iris – First Two Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

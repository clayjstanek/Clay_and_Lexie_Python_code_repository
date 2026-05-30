# -*- coding: utf-8 -*-
"""
Chapter 6 Cheat Sheet: NumPy Broadcasting

Goal for Lexie:
    Learn how NumPy does arithmetic between arrays of different shapes when
    the shapes are compatible. This is especially useful for standardizing data.
"""

import numpy as np


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# 1) Scalar broadcasting with a 1D array
# -----------------------------------------------------------------------------
section("1) Scalar broadcasting with a vector")

a = np.array([1, 2, 3])
b = 2

print("a =", a)
print("b =", b)
print("a + b =", a + b)
print("a * b =", a * b)


# -----------------------------------------------------------------------------
# 2) Scalar broadcasting with a 2D matrix
# -----------------------------------------------------------------------------
section("2) Scalar broadcasting with a matrix")

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("A =")
print(A)
print("A.shape =", A.shape)
print("A + 10 =")
print(A + 10)


# -----------------------------------------------------------------------------
# 3) Row-style broadcasting: add a 1D vector to each row
# -----------------------------------------------------------------------------
section("3) Add a vector to each row")

row_adjustment = np.array([10, 20, 30])

print("A.shape =", A.shape)
print("row_adjustment.shape =", row_adjustment.shape)
print("A + row_adjustment =")
print(A + row_adjustment)

# Why it works:
# A has shape              (2, 3)
# row_adjustment has shape    (3,)
# NumPy treats it like      (1, 3)
# Then it copies/broadcasts it across the rows.


# -----------------------------------------------------------------------------
# 4) Column-style broadcasting: add a column vector to each row position
# -----------------------------------------------------------------------------
section("4) Add a column vector down the rows")

col_adjustment = np.array([[100], [200]])

print("A.shape =", A.shape)
print("col_adjustment.shape =", col_adjustment.shape)
print("A + col_adjustment =")
print(A + col_adjustment)

# Why it works:
# A has shape              (2, 3)
# col_adjustment has shape (2, 1)
# The 1 column is stretched/broadcast across 3 columns.


# -----------------------------------------------------------------------------
# 5) A practical broadcasting example: standardize columns
# -----------------------------------------------------------------------------
section("5) Standardize feature columns")

X = np.array([
    [2, 75],
    [4, 82],
    [1, 60],
    [5, 90],
    [3, 78],
    [2, 65],
    [6, 92],
    [1, 58]
], dtype=float)

means = X.mean(axis=0)
stds = X.std(axis=0)

X_scaled = (X - means) / stds

print("X.shape =", X.shape)
print("means.shape =", means.shape)
print("stds.shape =", stds.shape)
print("means =", means)
print("stds =", stds)
print("X_scaled =")
print(np.round(X_scaled, 3))
print("Column means after scaling:", np.round(X_scaled.mean(axis=0), 6))
print("Column stds after scaling:", np.round(X_scaled.std(axis=0), 6))


# -----------------------------------------------------------------------------
# 6) Broadcasting failure example
# -----------------------------------------------------------------------------
section("6) Broadcasting failure example")

bad = np.array([1, 2, 3])
print("X.shape =", X.shape)
print("bad.shape =", bad.shape)

try:
    print(X + bad)
except ValueError as err:
    print("This failed, as expected.")
    print("Error message:", err)
    print("Reason: X has 2 columns, but bad has length 3.")

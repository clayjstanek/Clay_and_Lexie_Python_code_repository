# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:06:28 2026

@author: annil
"""

"""
Chapter 6 Homework Starter File
NumPy Array Broadcasting

Instructions:
    Fill in each TODO. Before doing arithmetic, print shapes.
    Broadcasting becomes much easier when you train yourself to inspect shapes.
"""

import numpy as np


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# Part A: Core Python practice
# =============================================================================
section("Chapter 6 Part A")

A = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# TODO 1: Add 5 to every element of A.
print('A + 5 = ', A + 5)
# TODO 2: Multiply every element of A by 2.
print('\nA * 2 =', A * 2)
row_adjustment = np.array([1, 2, 3])
# TODO 3: Add row_adjustment to A.
print('\nA + row_adjustment =', A + row_adjustment)
col_adjustment = np.array([[100], [200], [300]])
# TODO 4: Add col_adjustment to A.
print('\nA + col_adjustment =', A + col_adjustment)
# TODO 5: Print shapes of A, row_adjustment, and col_adjustment.
print('\nShape of A:', np.shape(A))
print('\nShape of row_adjustment:', np.shape(row_adjustment))
print('\nShape of col_adjustment:', np.shape(col_adjustment))
# TODO 6: In comments, explain why row_adjustment broadcasts across columns.
# row_adjustment broadcasts across across columns because the shape is (3,), so it fills in the rest of the rows since its missing two rows.

# TODO 7: In comments, explain why col_adjustment broadcasts down rows.
# col_adjustment broadcasts down rows because the shape is (3,1), so it fills in the rest of the columns since its missing two columns.

# =============================================================================
# Part B: Practical problem - standardizing features
# =============================================================================
section("Chapter 6 Part B")

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

# TODO 1: Compute the column means.
# means = ...
means = X.mean(axis=0)
print('Column means:', means)
# TODO 2: Compute the column standard deviations.
# stds = ...
stds = X.std(axis=0)
print('Column standard deviations:', stds)
# TODO 3: Standardize X.
# X_scaled = ...
X_scaled = (X - means) / stds
print('X_scaled:', X_scaled)
# TODO 4: Print X_scaled.mean(axis=0) and X_scaled.std(axis=0).
print('X_scaled mean:', X_scaled.mean(axis=0))
print('X_scaled standard deviation:', X_scaled.std(axis=0))
# TODO 5: In comments, explain why X - means works even though
# X.shape is (8, 2) and means.shape is (2,).
# X - means works because means.shape is broadcasted down each row so that it is subtracted from every row of X.

# Challenge: try adding a bad vector of length 3 to X.
# Wrap it in try/except so your script continues running.
bad = np.array([1, 2, 3])

try:
    print('X + bad:', X + bad)
except ValueError as error:
    print('Broadcasting error:', error)
# -*- coding: utf-8 -*-
"""
Chapter 5 Cheat Sheet: Index, Slice, and Reshape NumPy Arrays

Goal for Lexie:
    Learn how to access rows/columns of NumPy arrays, split a dataset into
    X and y, split train/test rows, and reshape 1D arrays into column vectors.

Run this file top-to-bottom, then experiment by changing the arrays.
"""

import numpy as np


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# 1) Create a 2D array from a list of lists
# -----------------------------------------------------------------------------
section("1) Create a 2D NumPy array")

data = np.array([
    [11, 22, 33],
    [44, 55, 66],
    [77, 88, 99]
])

print("data =")
print(data)
print("data.shape =", data.shape)  # (rows, columns)


# -----------------------------------------------------------------------------
# 2) Indexing: grab one element
# -----------------------------------------------------------------------------
section("2) Indexing single elements")

print("First row, first column: data[0, 0] =", data[0, 0])
print("Second row, third column: data[1, 2] =", data[1, 2])
print("Last row, last column: data[-1, -1] =", data[-1, -1])


# -----------------------------------------------------------------------------
# 3) Slicing: grab rows or columns
# -----------------------------------------------------------------------------
section("3) Slicing rows and columns")

print("All rows, first column: data[:, 0] =")
print(data[:, 0])

print("First row, all columns: data[0, :] =")
print(data[0, :])

print("All rows, all columns except last: data[:, :-1] =")
print(data[:, :-1])

print("All rows, last column: data[:, -1] =")
print(data[:, -1])


# -----------------------------------------------------------------------------
# 4) Machine learning pattern: split features X and target y
# -----------------------------------------------------------------------------
section("4) Split a table into X and y")

# Think of the last column as the value we want to predict.
X = data[:, :-1]  # all rows, all columns except the last one
y = data[:, -1]   # all rows, only the last column

print("X =")
print(X)
print("X.shape =", X.shape)

print("y =")
print(y)
print("y.shape =", y.shape)


# -----------------------------------------------------------------------------
# 5) Reshape y from 1D into a column vector
# -----------------------------------------------------------------------------
section("5) Reshape y into a column vector")

y_col = y.reshape((y.shape[0], 1))

print("y_col =")
print(y_col)
print("y_col.shape =", y_col.shape)


# -----------------------------------------------------------------------------
# 6) Train/test split by rows
# -----------------------------------------------------------------------------
section("6) Split rows into train and test sets")

split = 2
train = data[:split, :]   # rows 0 and 1
test = data[split:, :]    # rows 2 through end

print("train =")
print(train)
print("test =")
print(test)


# -----------------------------------------------------------------------------
# 7) Mini-practice: make your own array and try the same ideas
# -----------------------------------------------------------------------------
section("7) Your turn")

practice = np.array([
    [2, 75, 0],
    [4, 82, 1],
    [1, 60, 0],
    [5, 90, 1]
])

# TODO: Split practice into practice_X and practice_y.
# Hint: practice_X should contain the first two columns.
# Hint: practice_y should contain the last column.
practice_X = practice[:, :-1]
practice_y = practice[:, -1]

print("practice_X =")
print(practice_X)
print("practice_y =")
print(practice_y)

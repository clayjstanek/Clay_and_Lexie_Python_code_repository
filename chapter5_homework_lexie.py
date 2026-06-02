# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:50:35 2026

@author: annil
"""

"""
Chapter 5 Homework Starter File
Index, Slice, and Reshape NumPy Arrays

Instructions:
    Fill in each TODO. Run the file after each small section.
"""

import numpy as np


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# Part A: Core Python practice
# =============================================================================
section("Chapter 5 Part A")

data = np.array([
    [10, 2, 100],
    [20, 4, 200],
    [30, 6, 300],
    [40, 8, 400],
    [50, 10, 500],
    [60, 12, 600]
])

# TODO 1: Print the shape of data.
print('Shape of data:', np.shape(data))
# TODO 2: Print the first row.
print('\nFirst row:', data[0])
# TODO 3: Print the last row.
print('\nLast row:', data[-1])
# TODO 4: Print the first column.
print('\nFirst column:', data[:, 0])
# TODO 5: Print the last column.
print('\nLast column:', data[:, -1])
# TODO 6: Split data into X and y.
# X should be the first two columns.
# y should be the last column.
X = data[:, :2]
y = data[:, 2:]
print('\nX:', X)
print('y:', y)
# TODO 7: Print X.shape and y.shape.
print('\nShape of X:', X.shape)
print('Shape of y:', y.shape)
# TODO 8: Reshape y into a column vector called y_col.
y_col = y.reshape((y.shape[0], 1))
print('\ny_col:', y_col)
# TODO 9: Print y_col.shape.
print('\nShape of y_col:', y_col.shape)
# TODO 10: Split the first 4 rows into train and the last 2 rows into test.
train = data[:4]
test = data[4:]
print('\ntrain:', train)
print('test:', test)

# =============================================================================
# Part B: Practical problem - supervised learning table
# =============================================================================
section("Chapter 5 Part B")

students = np.array([
    [2, 75, 0],
    [4, 82, 1],
    [1, 60, 0],
    [5, 90, 1],
    [3, 78, 1],
    [2, 65, 0],
    [6, 92, 1],
    [1, 58, 0]
])

# Each row means:
# [hours_studied, practice_score, passed_final]

# TODO 1: Split students into X and y.
X = students[:, :2]
y = students[:, -1]
print('X:', X)
print('y:', y)
# TODO 2: Print the number of students.
# Hint: use X.shape[0]
print('\nNumber of students:', X.shape[0])
# TODO 3: Print the number of input features.
# Hint: use X.shape[1]
print('\nNumber of input features:', X.shape[1])
# TODO 4: Make y a column vector called y_col.
y_col = y.reshape((y.shape[0], 1))
print('y_col:', y_col)
# TODO 5: Use the first 6 students for training and the last 2 for testing.
# Create train_X, train_y, test_X, test_y.
train_X = X[:6]
train_y = y[:6]
test_X = X[6:]
test_y = y[6:]
# TODO 6: In a comment, explain why X is 2D but y often starts as 1D.
# X is 2D because it has multiple columns. y often starts as 1D because it is just one column, so each row can be represented as one number instead of its own row.

# =============================================================================
# Reflection Questions
# =============================================================================

# What does data[:,:-1] mean in plain English?
# data[:,:-1] means that you are basically just taking all of the rows of the data and all of the columns except the last one.

# What does data[:, -1] mean in plain English?
#data[:, -1] means that you are just taking only the last column with every row.

# Why is y.reshape((y.shape[0], 1)) useful?
# y.reshape((y.shape[0], 1)) is useful because it makes the data into a column vector, or a 2D vector, which can be easier to manipulate and do stuff with, and its more obvious that each number represents a different student.

# -*- coding: utf-8 -*-
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

# TODO 2: Print the first row.

# TODO 3: Print the last row.

# TODO 4: Print the first column.

# TODO 5: Print the last column.

# TODO 6: Split data into X and y.
# X should be the first two columns.
# y should be the last column.

# TODO 7: Print X.shape and y.shape.

# TODO 8: Reshape y into a column vector called y_col.

# TODO 9: Print y_col.shape.

# TODO 10: Split the first 4 rows into train and the last 2 rows into test.


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

# TODO 2: Print the number of students.
# Hint: use X.shape[0]

# TODO 3: Print the number of input features.
# Hint: use X.shape[1]

# TODO 4: Make y a column vector called y_col.

# TODO 5: Use the first 6 students for training and the last 2 for testing.
# Create train_X, train_y, test_X, test_y.

# TODO 6: In a comment, explain why X is 2D but y often starts as 1D.

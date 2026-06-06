# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:06:09 2026

@author: annil
"""

"""
Chapter 7 Homework Assignment

Vectors and Vector Arithmetic
"""

import numpy as np

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

# ========================================
# Part A - Python Skills Reinforcement
# ========================================

section('Part A - Python Skills Reinforcement')

# Create vectors a and b
a = np.array([2, 4, 6])
b = np.array([1, 3, 5])

# Print vectors a and b
print('\nVector a:', a)
print('Vector b:', b)

# Print a + b
print('\na + b =', a + b)

# Print a - b
print('\na - b =', a - b)

# Print a * b
print('\na * b =', a * b)

# Print a / b
print('\na / b =', a / b)

# Print a and b dot product
print('\ndot product of a and b =', a.dot(b))

# Print a * scalar 0.5
print('\na * 0.5 =', a * 0.5)

# Create vector c
c = [10, 20, 30]

# Print a + b + c
print('\na + b + c =', a + b + c)

# Manual dot product using for loop
def manual_dot_product(a, b):
    tot = 0 
    for i in range(len(a)):
        tot += a[i] * b[i]
    return tot

print('\nManual dot product =', manual_dot_product(a, b))

# Dot product vs element-wise multiplication explanation
"""
Element-wise multiplication and the dot product both multiply 
corresponding components of numpy arrays, but the difference is 
that the products are added together in a dot product so the output 
is a scalar. In element-wise multiplication the output is a numpy 
array with the amount of elements that you started with.
"""

# ========================================
# Part B - Practical Problem
# ========================================

section('Part B - Practical Problem')

# Students scores as vectors (Homework Score, Quiz Score, Project Score)
student_a = np.array([92, 85, 88])
student_b = np.array([80, 95, 90])
student_c = np.array([98, 70, 95])

# Print out student score arrays
print('\nStudent a scores:', student_a)
print('Student b scores:', student_b)
print('Student c scores:', student_c)

# Weights of each score
w = np.array([.20, .30, .50])
print('\nScore weights (Homework Score, Quiz Score, Project Score):', w)

# Weighted score for each student using dot product
a_score_weighted = student_a.dot(w)
b_score_weighted = student_b.dot(w)
c_score_weighted = student_c.dot(w)

# Print students weighted scores
print('\nStudent a weighted score:', a_score_weighted)
print('Student b weighted score:', b_score_weighted)
print('Student c weighted score:', c_score_weighted)

# Rank student weighted scores from highest to lowest
students = [
    ('student_a', a_score_weighted),
    ('student_b', b_score_weighted),
    ('student_c', c_score_weighted)
]
students_sorted = sorted(students, key=lambda x: x[1], reverse=True)
print('Rank of student weighted scores from highest to lowest:', [name for name, score in students_sorted])

# Why the dot product is a natural mathematical tool for combining measurements and weights
"""
The dot product just multiplies corresponding components of vectors and 
adds them together so for something like calculating weighted scores where 
we need to multiply each score by a fraction and add them all together, it 
is just the most efficient to use the dot product.
"""

# New weight vector with projust weight doubled
w_new = w * np.array([1, 1, 2])

# Normalize new vector
w_new = w_new / w_new.sum()
print('New score weights vector:', w_new)

# Recompute scores with new weights
a_score_weighted_new = student_a.dot(w_new)
b_score_weighted_new = student_b.dot(w_new)
c_score_weighted_new = student_c.dot(w_new)

# Print new weighetd scores
print('\nStudent a new weighted score:', a_score_weighted_new)
print('Student b new weighted score:', b_score_weighted_new)
print('Student c new weighted score:', c_score_weighted_new)

# Rerank student weighted scores from highest to lowest with new weights
students_new = [
    ('student_a', a_score_weighted_new),
    ('student_b', b_score_weighted_new),
    ('student_c', c_score_weighted_new)
]
students_sorted_new = sorted(students_new, key=lambda x: x[1], reverse=True)
print('Reranked student weighted scores from highest to lowest:', [name for name, score in students_sorted_new])

# Another real-world problem that could be represented with vectors and solved using a dot product
"""
Another real-world problem that could be represented with vectors and solved 
using a dot product is calculating total volume of rainfall over a certain 
period of time by having a vector with the average rate of rainfall of a bunch 
of storms over a period of time, and the other vector is duration of each storm.
"""

# ========================================
# Challenge Question
# ========================================

# How the student-ranking problem resembles what a ML model does when it combines features with learned weights
"""
The student-ranking problem resembles what a machine-learning model does when 
it combines features with learned weights because they both use a dot product 
to get a scalar to work with.
"""
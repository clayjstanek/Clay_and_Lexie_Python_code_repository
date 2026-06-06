# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:25:07 2026

@author: annil
"""

import math as math
import numpy as np

"""
Chapter 8 Homework: Vector Norms
"""
def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    
# ========================================
# Part A - Python Skills
# ========================================

section('Part A - Python Skills')

# Create vectors
vector_1 = np.array([1, 3, 5])
vector_2 = np.array([2, 4, 6])

# Print vectors 
print('\nvector_1:', vector_1)
print('vector_2:', vector_2)

# L1 norms of vectors
print('\nL1 norm vector_1:', np.linalg.norm(vector_1, ord=1))
print('L1 norm vector_2:', np.linalg.norm(vector_2, ord=1))

# L2 norms of vectors
print('\nL2 norm vector_1:', np.linalg.norm(vector_1, ord=2))
print('L2 norm vector_2:', np.linalg.norm(vector_2, ord=2))

# Max norms of vectors
print('\nL2 norm vector_1:', np.linalg.norm(vector_1, ord=np.inf))
print('L2 norm vector_2:', np.linalg.norm(vector_2, ord=np.inf))

# Manual L1 norm
def L1_norm(a):
    tot = 0
    for i in range(len(a)):
        tot += abs(a[i])
    return tot

# Print manual L1 norms
print('\nManual L1 norm for vector_1:', L1_norm(vector_1))
print('Manual L1 norm for vector_2:', L1_norm(vector_2))

# Manual L2 norm
def L2_norm(a):
    tot = 0
    for i in range(len(a)):
        tot += a[i] ** 2
    return math.sqrt(tot)

# Print manual L2 norms
print('\nManual L2 norm for vector_1:', L2_norm(vector_1))
print('Manual L2 norm for vector_2:', L2_norm(vector_2))

# ========================================
# Part B - Practical Problem
# ========================================

section('Part B - Practical Problem')

# Create vectors for three threat indicators: [Range Score, Velocity Score, Intent Score]
target_a = [8, 2, 1]
target_b = [4, 5, 4]
target_c = [2, 8, 7]

# Print target vectors
print('\ntarget_a vector:', target_a)
print('target_b vector:', target_b)
print('target_c vector:', target_c)

# L1, L2, and Max norms of each target
print('\nL1 norm of target_a:', np.linalg.norm(target_a, ord=1))
print('L2 norm for target_a:', np.linalg.norm(target_a, ord=2))
print('Max norm for target_a:', np.linalg.norm(target_a, ord=np.inf))

print('\nL1 norm of target_b:', np.linalg.norm(target_b, ord=1))
print('L2 norm for target_b:', np.linalg.norm(target_b, ord=2))
print('Max norm for target_b:', np.linalg.norm(target_b, ord=np.inf))

print('\nL1 norm of target_c:', np.linalg.norm(target_c, ord=1))
print('L2 norm for target_c:', np.linalg.norm(target_c, ord=2))
print('Max norm for target_c:', np.linalg.norm(target_c, ord=np.inf))

# Rank targets for each norm 
targets_L1_norms = [
    ('target_a', np.linalg.norm(target_a, ord=1)),
    ('target_b', np.linalg.norm(target_b, ord=1)),
    ('target_c', np.linalg.norm(target_c, ord=1))
]

targets_L2_norms = [
    ('target_a', np.linalg.norm(target_a, ord=2)),
    ('target_b', np.linalg.norm(target_b, ord=2)),
    ('target_c', np.linalg.norm(target_c, ord=2))
]

targets_max_norms = [
    ('target_a', np.linalg.norm(target_a, ord=np.inf)),
    ('target_b', np.linalg.norm(target_b, ord=np.inf)),
    ('target_c', np.linalg.norm(target_c, ord=np.inf))
]

L1_target_norms_sorted = sorted(targets_L1_norms, key=lambda x: x[1], reverse=True)
print('\nRank of student weighted scores from highest to lowest:', [target for target, norm in L1_target_norms_sorted])

L2_target_norms_sorted = sorted(targets_L2_norms, key=lambda x: x[1], reverse=True)
print('\nRank of student weighted scores from highest to lowest:', [target for target, norm in L2_target_norms_sorted])

max_target_norms_sorted = sorted(targets_max_norms, key=lambda x: x[1], reverse=True)
print('\nRank of student weighted scores from highest to lowest:', [target for target, norm in max_target_norms_sorted])

# Explain why different norms product different rankings
"""
Different norms product different rankings becasue the higher norms like L2 and 
max are closer to the largest components in the vectors.
"""

# Which norm might be most appropriate if larget values in any single dimension are especiall concerning
"""
I'm not really sure what this question is asking, but I would assume L2 norms 
would be appropriate since they are the most common. It might be max norms though 
since they are equal to the largest of any single dimension in the vector.
"""
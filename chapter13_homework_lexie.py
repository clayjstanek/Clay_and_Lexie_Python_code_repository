# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:57:30 2026

@author: annil
"""

import numpy as np

"""
Chapter 13 Homework - Tensors and Tensor Arithmetic
"""

def section(title):
    print('\n' +  '=' * 38 + '\n' + title + '\n' + '=' * 38)

# =======================================
# Part A - Python Skills Reinforcement
# =======================================

def partA_python_skills_reinforcement():
    section('Part A - Python Skills Reinforcement')

# Problem 1 - Creating Tensors
def problem1_creating_tensors():
    print('\nProblem 1 - Creating Tensors')

    T = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
        ])
    print('\nT:\n', T)
    
    # Shape of tensor T
    print('\nShape of tensor T:', T.shape)
    
    # Dimensions
    levels, rows, columns = T.shape
    print('\nNumber of levels:', levels)
    print('Number of rows:', rows)
    print('Number of columns:', columns)
    
    # Explanation
    print('\nThe tensor T has 3 dimensions. The first dimension is '
          'levels, the second is rows, and the third is columns.')
    
# Problem 2 - Tensor Indexing
def problem2_tensor_indexing():
    print('\nProblem 2 - Tensor Indexing')
    
    T = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
        ])
    
    # Print first level of T
    print('\nFirst level of T:\n', T[0])
    
    # Print second level of T
    print('\nSecond level of T:\n', T[1])
    
    # Print T[1, 0, 2]
    print('\nPrint T[1, 0, 2]:', T[1, 0 ,2])
    
    # Explanation
    print('\nThe index [1, 0, 2] is the second level, first '
          'row, and third column because tensors are indexed by '
          '[level, row, column].')
    
# Problem 3 - Tensor Arithmetic
def problem3_tensor_arithmetic():
    print('\nProblem 3 - Tensor Arithmetic')
    
    # Create tensors
    A = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
        ])
    
    B = np.array([
        [[10, 20], [30, 40]],
        [[50, 60], [70, 80]]
        ])
    
    print('\nA:\n', A)
    print('B:\n', B)
    
    print('\nA.shape =', A.shape)
    print('B.shape =', B.shape)
    
    # Print A + B
    print('\nA + B =\n', A + B)
    
    # Print A - B
    print('\nA - B =\n', A - B)
    
    # Print A * B
    print('\nA * B =\n', A * B)
    
    # Print A / B
    print('\nA / B =\n', A / B)
    
# Problem 4 - Manual Tensor Arithmetic
def problem4_manual_tensor_arithmetic():
    print('\nProblem 4 - Manual Tensor Arithmetic')
    
    A = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
        ])
    
    B = np.array([
        [[10, 20], [30, 40]],
        [[50, 60], [70, 80]]
        ])
    
    # Add A and B manually
    C_manual = np.zeros(A.shape, dtype=int)
    
    for level in range(A.shape[0]):
        for row in range(A.shape[1]):
            for col in range(A.shape[2]):
                C_manual[level, row, col] = A[level, row, col] + B[level, row, col]
    
    # Add A and B with numpy
    C_numpy = A + B
    
    # Compare C_manual and C_numpy
    print('\nC_manual:\n', C_manual)
    print('\nC_numpy:\n', C_numpy)
    
# Problem 5 - Tensor Product
def problem5_tensor_product():
    print('\nProblem 5 - Tensor Product')
    
    # Create tensors
    a = [1, 2, 3]
    b = [4, 5]
    
    print('\n a:', a)
    print('b:', b)
    
    # Tensor dot product
    print('\nTensor dot product of a and b:\n', np.tensordot(a, b, axes=0))
    
    # Shape of dot product
    print('\nShape of dot product:', np.tensordot(a, b, axes=0).shape)
    
    # Explanation
    print('\nThe shape is a matrix because when you multiply a '
          'vector of length 3 and a vector of length 2, it creates '
          'a 3 by 2 matrix.')
    
# =================================================
# Part b - Practical Problem: Image Cube Analysis
# =================================================

def partB_practical_problem():
    section('Part B - Practical Problem: Image Cube Analysis')
    
    Video = np.array([
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
        [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
        [[40, 41, 42], [43, 44, 45], [46, 47, 48]]
        ])
    
    print('\nVideo.shape =', Video.shape, '\n')
    
    # Average brightness of each frame
    for frame in range(Video.shape[0]):
        print('Average brightness of frame' , frame , ':' , np.mean(Video[frame]))
        
    # Average brighness of entire video
    print('\nAverage brightness of video:', np.mean(Video))
    
    # Brightest pixel
    print('\nBrightest pixel value:', Video.max())
    print('Frame with brightest pixel:', np.unravel_index(np.argmax(Video), Video.shape)[0])
    
    # Increase brightness
    Video_brightened = Video * 1.25
    print('\nBrightened video:', Video_brightened)
    
    # Explanation
    print('Videos naturally require tensors rather than matrices '
          'because matrices only have 2 dimensions, and you would '
          'only be able to represent one frame of a video with a '
          '2d tensor. In order to store multiple frames, you need '
          'another dimension.')
    
# Challenge Question
def challenge_question():
    print('The batch dimension represents the amount of samples. '
          'In this case, it represents the number of images. '
          'Channels represent different features like intensity '
          'of red green and blue in an image, or just information '
          'at each pixel. Height and width represent vertical and '
          'horizontal pixel dimensions of each image, respectively.')


def main():
    partA_python_skills_reinforcement()
    problem1_creating_tensors()
    problem2_tensor_indexing()
    problem3_tensor_arithmetic()
    problem4_manual_tensor_arithmetic()
    problem5_tensor_product()
    partB_practical_problem()
    challenge_question()
    
if __name__ == '__main__':
    main()
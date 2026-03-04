# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:20:21 2026
For Lexie to be able to inspect an Python package/library and see what it contains for methods and help
@author: cstanek
"""

# cheatsheet_exploring_libraries.py
# How to explore any Python library independently.
# Run this file, then try things interactively in the console.

print("\n=== HOW TO EXPLORE A PYTHON LIBRARY ===\n")

# ---------------------------------------
# 1) Import the library
# ---------------------------------------

import math
import random

print("Imported math and random.\n")


# ---------------------------------------
# 2) See what's inside a library: dir()
# -------------------------sdd
print("\nNotice all the function names listed.\n")

print('\n\n Math library \n\n')

print(dir(math))
print('\n\n Random library \n\n')
print(dir(random))
# ---------------------------------------
# 3) Read documentation: help()
# ---------------------------------------

print("Use help(math.sqrt) to read about sqrt:\n")
help(math.sqrt)

# Tip:
# In Spyder or IPython console, you can also type:
#   math.sqrt?
# to see documentation quickly.


# ---------------------------------------
# 4) Look at the docstring directly
# ---------------------------------------

print("\nEvery function has a __doc__ string:\n")
print(math.sqrt.__doc__)
print()


# ---------------------------------------
# 5) Inspect the type of something
# ---------------------------------------

x = math.sqrt(16)
print("math.sqrt(16) =", x)
print("type of x =", type(x))
print()


# ---------------------------------------
# 6) Discover function arguments
# ---------------------------------------

print("What arguments does random.randint expect?")
help(random.randint)
print()

print("What arguments does random.gauss expect?")
help(random.gauss)
print()


# ---------------------------------------
# 7) Explore objects dynamically
# ---------------------------------------

sample_list = [1, 2, 3]
print("dir(sample_list):\n")
print(dir(sample_list))
print("\nThese are all methods available for lists.\n")

# Try one:
print("sample_list.append.__doc__:\n")
print(sample_list.append.__doc__)
print()


# ---------------------------------------
# 8) Use tab completion (important!)
# ---------------------------------------

print("In Spyder/IPython:")
print("Type: math.")
print("Then press TAB to see available methods.\n")


# ---------------------------------------
# 9) A Practical Exploration Pattern
# ---------------------------------------

print("A good workflow when exploring a new library:\n")
print("1) import library")
print("2) dir(library)")
print("3) help(library.some_function)")
print("4) Try a small example")
print("5) Inspect output with type()")
print("6) Modify inputs and observe behavior\n")


# ---------------------------------------
# 10) Mini Exercise
# ---------------------------------------

print("Mini exercise:")
print("1) Use dir(random) to find something interesting.")
print("2) Use help() on it.")
print("3) Try calling it.")
print()

print("You're now equipped to explore ANY Python library.\n")

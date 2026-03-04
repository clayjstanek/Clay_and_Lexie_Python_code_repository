# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:28:05 2026
For Lexie to explore key methods in library:  math
@author: cstan
"""

# cheatsheet_math.py
# A compact, runnable cheat sheet for Python's built-in math library.
# Run this file in Spyder (or any Python environment) and read the comments/output.

import math

print("\n=== math cheat sheet ===\n")

# ----------------------------
# Constants
# ----------------------------
print("Constants")
print("math.pi  =", math.pi)
print("math.e   =", math.e)
print("math.tau =", math.tau)     # 2*pi
print("math.inf =", math.inf)
print("math.nan =", math.nan)
print()

# ----------------------------
# Basic arithmetic helpers
# ----------------------------
print("Basic helpers")
x = 3.7
print("x =", x)
print("math.floor(x) =", math.floor(x))   # largest integer <= x
print("math.ceil(x)  =", math.ceil(x))    # smallest integer >= x
print("math.trunc(x) =", math.trunc(x))   # drop fractional part toward 0
print("abs(-5)       =", abs(-5))         # built-in, but used constantly
print()

# ----------------------------
# Powers, roots, exponentials, logs
# ----------------------------
print("Powers / roots / exp / logs")
a = 9
print("math.sqrt(9)    =", math.sqrt(a))
print("math.pow(2, 5)  =", math.pow(2, 5))   # returns float
print("2**5            =", 2**5)             # Python operator, returns int here
print("math.exp(1)     =", math.exp(1))      # e^1
print("math.log(8, 2)  =", math.log(8, 2))   # log base 2
print("math.log10(100) =", math.log10(100))
print("math.log2(1024) =", math.log2(1024))
print()

# ----------------------------
# Trig (radians) + degree conversions
# ----------------------------
print("Trig (angles are in radians)")
deg = 30
rad = math.radians(deg)
print(f"{deg} degrees in radians =", rad)
print("sin(30°) =", math.sin(rad))
print("cos(30°) =", math.cos(rad))
print("tan(30°) =", math.tan(rad))
print("Convert back to degrees:", math.degrees(rad))
print()

# ----------------------------
# Helpful geometry
# ----------------------------
print("Geometry helpers")
print("math.hypot(3,4) =", math.hypot(3, 4))   # sqrt(3^2 + 4^2)
print()

# ----------------------------
# Factorials and combinations (great for probability)
# ----------------------------
print("Counting (probability / combinatorics)")
n = 6
k = 2
print("math.factorial(6) =", math.factorial(n))
print("math.comb(6,2)    =", math.comb(n, k))  # n choose k
print("math.perm(6,2)    =", math.perm(n, k))  # nPk
print()

# ----------------------------
# Floating-point comparisons (important!)
# ----------------------------
print("Floating-point comparisons")
a = 0.1 + 0.2
b = 0.3
print("0.1 + 0.2 =", a)
print("0.3       =", b)
print("a == b    =", a == b)
print("math.isclose(a, b) =", math.isclose(a, b))
print()

# ----------------------------
# Special functions you’ll see often
# ----------------------------
print("Special functions")
print("math.gcd(84, 30) =", math.gcd(84, 30))   # greatest common divisor
print("math.fsum([0.1]*10) =", math.fsum([0.1]*10))  # more accurate sum
print("\nDone.\n")

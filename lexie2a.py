# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:06:31 2026

@author: cstan

Today:

Variables
Control flow
Functions
Lists

Next lesson:

Dictionaries
Files
Basic plotting
Maybe simple probability simulation

And you are laying groundwork for:

Linear algebra
Monte Carlo methods
Neural networks
"""

""" Variables and Types
Everything in Python is an object.

Variables are just labels pointing to objects.

Types matter — especially later in ML.
"""
#hi my name is Lexie

x = 5
y = 3.2
name = "Lexie"
flag = True

print(type(x))
print(type(y))
print(type(name))
print(f'Type: {type(flag)}')
print('\n')

"""
Introduce:

Integer division, Modulus, Exponentiation

This builds number intuition you’ll use later for probability and linear algebra.

"""

a = 10
b = 3

print(f'a+b = {a + b}')
print(a / b)
print(a // b)
print(a % b)
print(a ** b)
print(f'pow(a,b) = {pow(a,b)}')

a = 9
print('\n\n')
print(f'a // b = {a // b}')
print(f'a%b = {a%b}')


"""
conditionals
Key Concept-- Indentation defines logic blocks. Must use colon character!
"""

age = 17

if age >= 18:
    print("Adult")
else:
    print("Minor")
    
score = 87

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("Below B")
    
"""
loops
"""

for i in range(5):
    print(i)
    
for i in range(1, 11):
    print(i**2)


"""
lists
"""

numbers = [1, 2, 3, 4, 5]
print(numbers[0])
print(numbers[-1])

numbers.append(6)
numbers.remove(3)

for num in numbers:
    print(num * 2)

squares = [x**2 for x in range(10)]

"""
Functions
Key Concept

Functions encapsulate logic.

This is how large systems are built.

Every ML model is just a very complicated function.
"""

def greet(name):
    print("Hello", name)


greet("Lexie")

def square(x):
    return x**2

result = square(7)
print(result)

#mini project
def win_percentage(wins, losses):
    total = wins + losses
    return (wins / total)

w = int(input("Enter wins: "))
l = int(input("Enter losses: "))

print("Win %:", win_percentage(w, l))
print("\n\n")

import random

secret = random.randint(1, 20)
guess = 0

while guess != secret:
    guess = int(input("Guess a number between 1 and 20: "))
    
    if guess < secret:
        print("Too low!")
    elif guess > secret:
        print("Too high!")
    else:
        print("You got it!")

"""
arrays with numpy
Why does that work differently than regular lists?
"""
print("\n\n")

import numpy as np

a = [1,2,3]
b = [4,5,6]
print(f'a+b = {a+b}')

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f'a+b = {a + b}')

print(np.dot(a,b))
print(np.cross(a,b))








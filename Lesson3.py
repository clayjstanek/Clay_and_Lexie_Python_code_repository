# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 19:07:50 2026

@author: cstan
"""

"""
ig Theme: “Organizing Information”

Right now you know:

Lists = ordered sequences

NumPy arrays = mathematical sequences

Now learn:

Dictionaries = labeled data

JSON = portable dictionaries

Pandas = labeled 2D data

This creates a beautiful hierarchy:

list → dict → JSON → DataFrame
"""

student = {
    "name": "Lexie",
    "age": 17,
    "favorite_sport": "Tennis",
    "gpa": 3.9
}

print(student["name"])

student["gpa"] = 4.0
student["graduation_year"] = 2027

for key, value in student.items():
    print(key, value)

"""
Concept to Emphasize

Lists are indexed by position.

Dictionaries are indexed by labels.

That distinction is huge.

Lexie:

When would you use a list vs a dictionary?

That question builds design thinking.
"""

# Nested data
classroom = {
    "student1": {"name": "Lexie", "gpa": 4.0},
    "student2": {"name": "Alex", "gpa": 3.5}
}

print(classroom["student1"]["gpa"])

#Part 3 – JSON (15 min, light)
import json

json_string = json.dumps(student)
print(json_string)


data = json.loads(json_string)
print(type(data))

with open("student.json", "w") as f:
    json.dump(student, f)
    
with open("student.json", "r") as f:
    loaded_student = json.load(f)

#Part 4 – Files (10–15 min)

with open("hello.txt", "w") as f:
    f.write("Hello Lexie")

with open("hello.txt", "r") as f:
    content = f.read()

print(content)
"""
“w” overwrites

“a” appends

“r” reads

This is foundational.
"""

#Part 5 – Gentle Pandas Intro (20 min).  pandas is for making dataframes
# dataframes is another word for a 'table' of data in memory that can be addressed
#recalled, and modified.  Pandas is one of the most powerful libraries in python
# and must be installed into your environment to work.  Such as:
#   pip install pandas <return>
# from your conda prompt (make sure your environment is set to Lexie before doing that step!)


import pandas as pd

df = pd.read_csv("name_age_gpa.csv")
print(df)

"""
Then in Spyder:
Open Variable Explorer and click the DataFrame.

see:

Rows
Columns
Headers
"""
print(df["gpa"])
print(df["gpa"].mean())

import random

values = []

for _ in range(1000):
    r = random.random()
    values.append(r)

print(values[:10])   # show first 10

#  using the random library as a simple example

import random

wins = 0

for _ in range(1000):
    if random.random() < 0.6:
        wins += 1

print("Estimated win rate:", wins / 1000)

#Part 6 – Visualizing Randomness with Matplotlib
## must do a pip install matplotlib <return> in conda prompt for this import to work.
import matplotlib.pyplot as plt

plt.hist(values, bins=20)
plt.title("Histogram of random.random()")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

"""
Why isn’t it perfectly flat?

That opens the door to:

Sampling variability

Law of Large Numbers

Noise vs signal

Then change sample size:
"""

values = []

for _ in range(100000):
    values.append(random.random())

plt.hist(values, bins=20)
plt.show()

"""
You’ve now connected:

Loops

Lists

Random number generation

Probability

Visualization

Sampling behavior

That is essentially:

Introductory computational statistics.

At age 17.
"""

averages = []
total = 0

for i in range(1, 1001):
    r = random.random()
    total += r
    averages.append(total / i)

plt.plot(averages)
plt.title("Running Average of random.random()")
plt.show()

#Part 7 – Lists vs NumPy Arrays

import numpy as np

np_values = np.array(values)

print(type(values))
print(type(np_values))

plt.hist(np_values, bins=20)
plt.title("Histogram using NumPy array")
plt.show()


print(values + values)
print(np_values + np_values)

print(np_values.mean())

print(sum(values) / len(values))
#which mean calculation is easier to write and understand?

"""
Now she sees:

Same random numbers

Transformed distribution

Non-uniform shape emerges

This connects math to visualization.
"""

squared = np_values ** 2

plt.hist(squared, bins=20)
plt.title("Squared random values")
plt.show()










# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:24:37 2026
Lexie study sheet for using the library:  random
@author: cstanek
"""

# cheatsheet_random.py
# A runnable cheat sheet for Python's built-in random library, with simple plots.
# Run this file and tweak parameters (N, bins, distribution settings) to explore.

import random
import matplotlib.pyplot as plt

print("\n=== random cheat sheet (with plots) ===\n")

# A helper to sample a function N times and return a list
def sample_many(fn, n=5000):
    return [fn() for _ in range(n)]

# A helper to make a histogram quickly
def show_hist(data, bins=30, title="Histogram"):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

# ----------------------------
# Core “most used” functions
# ----------------------------
print("Most used:")
print("random.random() -> uniform float in [0,1)")
print("random.randint(a,b) -> integer in [a,b] (inclusive)")
print("random.choice(seq) -> one random element")
print("random.sample(seq,k) -> k unique elements (no replacement)")
print("random.shuffle(list) -> shuffle a list in-place")
print()

# Optional: set a seed for repeatable results (same random sequence each run)
random.seed(42)
print("Seed set to 42 (repeatable demo).")
print()

# ----------------------------
# 1) Uniform(0,1): random.random()
# ----------------------------
N = 10000
u = sample_many(random.random, n=N)
print(f"Generated {N} samples from random.random().")
show_hist(u, bins=25, title="Uniform samples from random.random() in [0,1)")

# ----------------------------
# 2) Uniform(a,b): random.uniform(a,b)
# ----------------------------
a, b = -2, 3
u2 = sample_many(lambda: random.uniform(a, b), n=N)
show_hist(u2, bins=25, title=f"Uniform samples from random.uniform({a},{b})")

# ----------------------------
# 3) Normal (Gaussian): random.gauss(mu, sigma)
# ----------------------------
mu, sigma = 0, 1
g = sample_many(lambda: random.gauss(mu, sigma), n=N)
show_hist(g, bins=40, title=f"Normal samples from random.gauss(mu={mu}, sigma={sigma})")

# ----------------------------
# 4) Exponential: random.expovariate(lambd)
# lambd is the RATE (λ). Mean = 1/λ.
# ----------------------------
lambd = 1.5
e = sample_many(lambda: random.expovariate(lambd), n=N)
show_hist(e, bins=40, title=f"Exponential samples from random.expovariate(λ={lambd})")

# ----------------------------
# 5) Discrete outcomes: randint + choice
# ----------------------------
dice = sample_many(lambda: random.randint(1, 6), n=N)
show_hist(dice, bins=6, title="Dice rolls using random.randint(1,6)")

colors = ["red", "green", "blue"]
picked = sample_many(lambda: random.choice(colors), n=30)
print("random.choice(colors) examples:", picked)
print()

# ----------------------------
# 6) sample vs choices (replacement vs no replacement)
# ----------------------------
cards = list(range(1, 11))  # pretend "cards" numbered 1..10
print("cards:", cards)
print("random.sample(cards, 4) (no replacement):", random.sample(cards, 4))
print("random.choices(cards, k=4) (with replacement):", random.choices(cards, k=4))
print()

# ----------------------------
# 7) Shuffle demo
# ----------------------------
deck = list(range(1, 11))
random.shuffle(deck)
print("Shuffled list (in-place):", deck)
print("\nDone.\n")

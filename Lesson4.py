# -*- coding: utf-8 -*-
"""
Lesson 4 for Lexie: From Raw Data to Insight (Dictionaries -> Pandas DataFrames)

What this lesson covers (in order):
1) Quick bridge: list-of-dicts (Python) -> DataFrame (pandas)
2) Load Titanic dataset (titanic.csv) into a DataFrame
3) Inspect: head(), columns, shape, info(), describe()
4) Select columns (Series vs DataFrame)
5) Filter rows (boolean masks)
6) GroupBy summaries (the big concept!)
7) Missing data: isnull(), fillna()
8) Basic plots using matplotlib
9) Save a cleaned dataset to disk

How to run:
- Put this file in the same folder as titanic.csv, then run in Spyder.
- If you don't have titanic.csv yet, the script will fall back to a tiny demo dataset
  so the lesson still runs end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def try_load_titanic_csv(csv_path: Path) -> pd.DataFrame:
    """
    Attempts to load a Titanic CSV at csv_path.
    If not found, returns a small built-in demo DataFrame with Titanic-like columns
    so the lesson can still run.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df

    # Fallback: small demo dataset (enough to practice the concepts)
    section("NOTE: titanic.csv not found — using a small demo dataset instead")
    data = [
        {"PassengerId": 1, "Survived": 0, "Pclass": 3, "Name": "Braund, Mr. Owen", "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
        {"PassengerId": 2, "Survived": 1, "Pclass": 1, "Name": "Cumings, Mrs. John", "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.2833, "Embarked": "C"},
        {"PassengerId": 3, "Survived": 1, "Pclass": 3, "Name": "Heikkinen, Miss. Laina", "Sex": "female", "Age": 26, "SibSp": 0, "Parch": 0, "Fare": 7.925, "Embarked": "S"},
        {"PassengerId": 4, "Survived": 1, "Pclass": 1, "Name": "Futrelle, Mrs. Jacques", "Sex": "female", "Age": 35, "SibSp": 1, "Parch": 0, "Fare": 53.1, "Embarked": "S"},
        {"PassengerId": 5, "Survived": 0, "Pclass": 3, "Name": "Allen, Mr. William", "Sex": "male", "Age": None, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"},
        {"PassengerId": 6, "Survived": 0, "Pclass": 3, "Name": "Moran, Mr. James", "Sex": "male", "Age": None, "SibSp": 0, "Parch": 0, "Fare": 8.4583, "Embarked": "Q"},
    ]
    return pd.DataFrame(data)


# ----------------------------
# Lesson Script
# ----------------------------
def part_1_bridge_list_of_dicts_to_dataframe() -> None:
    section("Part 1 — Bridge: list-of-dicts -> DataFrame")

    students = [
        {"name": "Lexie", "gpa": 4.0, "sport": "Tennis"},
        {"name": "Alex", "gpa": 3.5, "sport": "Soccer"},
        {"name": "Jordan", "gpa": 3.8, "sport": "Tennis"},
    ]

    print("Python data (list of dictionaries):")
    print(students)

    # Compute average GPA using plain Python
    total = 0
    for s in students:
        total += s["gpa"]
    print("\nAverage GPA (plain Python) =", total / len(students))

    # Now the same data in pandas
    df_students = pd.DataFrame(students)
    print("\nSame data as a DataFrame:")
    print(df_students)

    print("\nAverage GPA (pandas) =", df_students["gpa"].mean())


def part_2_load_and_inspect(df: pd.DataFrame) -> None:
    section("Part 2 — Load + Inspect Titanic DataFrame")

    print("First 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(list(df.columns))

    print("\nShape (rows, cols):", df.shape)

    print("\nInfo (types + missing counts):")
    # df.info() prints directly
    df.info()

    # Describe numeric columns
    print("\nDescribe (numeric columns):")
    print(df.describe())


def part_3_select_and_filter(df: pd.DataFrame) -> None:
    section("Part 3 — Selecting Columns and Filtering Rows")

    # Column selection: Series vs DataFrame
    print("Selecting one column returns a Series:")
    if "Age" in df.columns:
        age_series = df["Age"]
        print(type(age_series))
        print(age_series.head())

    print("\nSelecting multiple columns returns a DataFrame:")
    cols = [c for c in ["Age", "Fare", "Survived", "Pclass", "Sex"] if c in df.columns]
    df_subset = df[cols]
    print(type(df_subset))
    print(df_subset.head())

    # Filtering rows
    if "Survived" in df.columns:
        survivors = df[df["Survived"] == 1]
        nonsurvivors = df[df["Survived"] == 0]
        print("\nRows where Survived == 1:", survivors.shape[0])
        print("Rows where Survived == 0:", nonsurvivors.shape[0])

        if "Age" in df.columns:
            print("\nMean Age (survivors):", survivors["Age"].mean())
            print("Mean Age (non-survivors):", nonsurvivors["Age"].mean())


def part_4_groupby(df: pd.DataFrame) -> None:
    section("Part 4 — GroupBy (Big Concept!)")

    # Survival rate overall
    if "Survived" in df.columns:
        print("Overall survival rate:", df["Survived"].mean())

    # Survival rate by sex
    if "Sex" in df.columns and "Survived" in df.columns:
        print("\nSurvival rate by Sex:")
        print(df.groupby("Sex")["Survived"].mean())

    # Survival rate by passenger class
    if "Pclass" in df.columns and "Survived" in df.columns:
        print("\nSurvival rate by Pclass:")
        print(df.groupby("Pclass")["Survived"].mean())

    # Two-way groupby: Sex + Pclass
    if all(c in df.columns for c in ["Sex", "Pclass", "Survived"]):
        print("\nSurvival rate by Sex AND Pclass:")
        print(df.groupby(["Sex", "Pclass"])["Survived"].mean())


def part_5_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    section("Part 5 — Missing Data (isnull + fillna)")

    df2 = df.copy()

    if "Age" in df2.columns:
        missing_age = df2["Age"].isnull().sum()
        print("Missing Age count:", missing_age)

        if missing_age > 0:
            age_mean = df2["Age"].mean()
            print("Filling missing Age values with mean Age =", age_mean)
            df2["Age"] = df2["Age"].fillna(age_mean)

            print("Missing Age count after fill:", df2["Age"].isnull().sum())

    return df2


def part_6_plots(df: pd.DataFrame) -> None:
    section("Part 6 — Basic Plots (matplotlib)")

    # Histogram of Age
    if "Age" in df.columns:
        plt.figure()
        df["Age"].hist(bins=30)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

    # Bar plot: survival rate by Sex
    if all(c in df.columns for c in ["Sex", "Survived"]):
        plt.figure()
        df.groupby("Sex")["Survived"].mean().plot(kind="bar")
        plt.title("Survival Rate by Sex")
        plt.xlabel("Sex")
        plt.ylabel("Survival Rate")
        plt.show()

    # Bar plot: average Fare by Pclass
    if all(c in df.columns for c in ["Pclass", "Fare"]):
        plt.figure()
        df.groupby("Pclass")["Fare"].mean().plot(kind="bar")
        plt.title("Average Fare by Passenger Class")
        plt.xlabel("Pclass")
        plt.ylabel("Average Fare")
        plt.show()


def part_7_save_cleaned(df: pd.DataFrame, out_path: Path) -> None:
    section("Part 7 — Save a cleaned dataset")

    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to: {out_path}")


def main() -> None:
    # Ensure script runs relative to its own folder (useful in Spyder)
    try:
        os.chdir(Path(__file__).resolve().parent)
    except Exception:
        pass

    section("Lesson 4 — From Raw Data to Insight")

    # 1) Bridge
    part_1_bridge_list_of_dicts_to_dataframe()

    # 2) Load Titanic
    csv_path = Path("titanic.csv")
    df = try_load_titanic_csv(csv_path)

    # 3) Inspect
    part_2_load_and_inspect(df)

    # 4) Select + Filter
    part_3_select_and_filter(df)

    # 5) GroupBy
    part_4_groupby(df)

    # 6) Missing data (Age)
    df_clean = part_5_missing_data(df)

    # 7) Plots
    part_6_plots(df_clean)

    # 8) Save cleaned dataset
    out_path = Path("titanic_cleaned.csv")
    part_7_save_cleaned(df_clean, out_path)

    section("End of Lesson 4")
    print("Next: You'll use these same skills in a short mini-project.")


if __name__ == "__main__":
    main()

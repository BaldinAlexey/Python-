import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_wine_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, delimiter=';')

    df["quality_category"] = pd.cut(
        df["quality"],
        bins=[0, 4, 6, 10],
        labels=["Низкое", "Среднее", "Высокое"]
    )
    return df


def describe_quality(df):
    return df["quality"].describe()


def detect_outliers(df):
    numeric = df.select_dtypes(include=["number"])
    outlier_mask = (numeric < numeric.quantile(0.01)) | (numeric > numeric.quantile(0.99))
    return outlier_mask.sum()



def correlation_matrix(df):
    return df.corr(numeric_only=True)


def compare_composition(df):
    return df.groupby("quality_category").mean(numeric_only=True)


def acidity_vs_quality(df):
    return df.groupby("quality")["fixed acidity"].mean()


def alcohol_vs_quality(df):
    return df.groupby("quality")["alcohol"].mean()



from scipy.stats import ttest_ind, pearsonr


def sugar_effect(df):
    low = df[df["quality_category"] == "Низкое"]["residual sugar"]
    high = df[df["quality_category"] == "Высокое"]["residual sugar"]
    return ttest_ind(low, high, equal_var=False)


def ph_vs_acidity(df):
    return pearsonr(df["pH"], df["fixed acidity"])


def alcohol_quality_stats(df):
    groups = [
        df[df["quality_category"] == "Низкое"]["alcohol"],
        df[df["quality_category"] == "Среднее"]["alcohol"],
        df[df["quality_category"] == "Высокое"]["alcohol"],
    ]
    return ttest_ind(groups[0], groups[2], equal_var=False)



def plot_distributions(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(x="quality", data=df)
    plt.title("Распределение качества вина")
    plt.show()


def plot_correlations(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Матрица корреляций")
    plt.show()


def plot_alcohol(df):
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="quality_category", y="alcohol", data=df)
    plt.title("Алкоголь по категориям качества")
    plt.show()


def main():
    df = load_wine_data()

    print("\n--- ОПИСАНИЕ КАЧЕСТВА ---")
    print(describe_quality(df))

    print("\n--- ВЫБРОСЫ ---")
    print(detect_outliers(df))

    print("\n--- КОРРЕЛЯЦИИ ---")
    print(correlation_matrix(df))

    print("\n--- СРАВНЕНИЕ СОСТАВА ---")
    print(compare_composition(df))

    print("\n--- КИСЛОТНОСТЬ vs КАЧЕСТВО ---")
    print(acidity_vs_quality(df))

    print("\n--- АЛКОГОЛЬ vs КАЧЕСТВО ---")
    print(alcohol_vs_quality(df))

    print("\n--- ГИПОТЕЗА: САХАР ВЛИЯЕТ НА КАЧЕСТВО ---")
    print(sugar_effect(df))

    print("\n--- ГИПОТЕЗА: pH СВЯЗАН С КИСЛОТНОСТЬЮ ---")
    print(ph_vs_acidity(df))

    print("\n--- ГИПОТЕЗА: РАЗНИЦА В АЛКОГОЛЕ ---")
    print(alcohol_quality_stats(df))

    plot_distributions(df)
    plot_correlations(df)
    plot_alcohol(df)


if __name__ == "__main__":
    main()
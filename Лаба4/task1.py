"""
Задание 1: Сравнительный анализ NumPy и чистого Python
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

sns.set(style="whitegrid")


def generate_test_datasets():
    """Создаёт 4 набора данных разных размеров."""
    return {
        "small": np.random.random(1000),
        "medium": np.random.random(10000),
        "large": np.random.random(100000),
        "xlarge": np.random.random(1000000),
    }


# Чистый Python
def py_square(data):
    return [x ** 2 for x in data]


def py_sin(data):
    return [math.sin(x) for x in data]


def py_sum(data):
    total = 0
    for x in data:
        total += x
    return total


def py_max(data):
    max_val = data[0]
    for x in data:
        if x > max_val:
            max_val = x
    return max_val


# NumPy версии
def np_square(data):
    return np.square(data)


def np_sin(data):
    return np.sin(data)


def np_sum(data):
    return np.sum(data)


def np_max(data):
    return np.max(data)


def benchmark(op1, op2, data):
    """Сравнивает время выполнения операций в чистом Python и NumPy."""
    py_data = data.tolist()

    start = time.time()
    op1(py_data)
    py_time = time.time() - start

    start = time.time()
    op2(data)
    np_time = time.time() - start

    return py_time, np_time


def plot_results(results):
    """Строит графики времени и ускорения."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 5))

    datasets = list(results.keys())
    ops = list(next(iter(results.values())).keys())

    for op in ops:
        py_times = [results[d][op][0] for d in datasets]
        np_times = [results[d][op][1] for d in datasets]
        plt.plot(datasets, py_times, "o--", label=f"{op} (Python)")
        plt.plot(datasets, np_times, "s--", label=f"{op} (NumPy)")

    plt.title("Сравнение времени выполнения операций")
    plt.xlabel("Размер набора данных")
    plt.ylabel("Время (сек)")
    plt.legend()
    plt.show()

    # Тепловая карта ускорения
    import pandas as pd
    speedup = pd.DataFrame(
        {
            d: {
                op: round(results[d][op][0] / results[d][op][1], 2)
                for op in results[d]
            }
            for d in results
        }
    )
    sns.heatmap(speedup, annot=True, cmap="viridis")
    plt.title("Ускорение NumPy над Python (x раз)")
    plt.show()


def main():
    datasets = generate_test_datasets()
    operations = {
        "square": (py_square, np_square),
        "sin": (py_sin, np_sin),
        "sum": (py_sum, np_sum),
        "max": (py_max, np_max),
    }

    results = {}
    for name, data in datasets.items():
        results[name] = {}
        for op_name, (py_op, np_op) in operations.items():
            results[name][op_name] = benchmark(py_op, np_op, data)
            print(f"{name:<8} {op_name:<6} -> Python: {results[name][op_name][0]:.4f}s, NumPy: {results[name][op_name][1]:.4f}s")

    plot_results(results)


if __name__ == "__main__":
    main()

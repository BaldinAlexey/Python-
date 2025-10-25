"""
task3_metrics.py
Задание 3: Сравнительный анализ и метрики для алгоритма муравьиной оптимизации
"""

import time
import random
from functools import wraps, lru_cache, reduce
from statistics import mean
from task2 import generate_points, ant_colony_optimization  # используем предыдущую реализацию

# ===========================
# 1. Декоратор для измерения времени выполнения
# ===========================
def timing_decorator(func):
    """
    Декоратор: измеряет время выполнения функции
    и выводит его в консоль.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper


# ===========================
# 2. Пример использования lru_cache для мемоизации
# ===========================
@lru_cache(maxsize=None)
def heavy_calculation(n):
    """
    Пример "тяжёлой" функции — здесь просто
    моделируем нагрузку через вычисления.
    lru_cache запоминает результаты и повторно не считает.
    """
    random.seed(n)
    total = 0
    for _ in range(200000):
        total += random.random()
    return total


# ===========================
# 3. Композиция функций (pipeline)
# ===========================
def compose(*functions):
    """
    compose(f, g, h)(x) = f(g(h(x)))
    Позволяет соединять функции в одну цепочку.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


# Пример функций для pipeline анализа
def normalize_data(results):
    """
    results: список длин маршрута
    нормализация к диапазону [0, 1]
    """
    if not results:
        return []
    min_val = min(results)
    max_val = max(results)
    if max_val == min_val:
        return [0.0 for _ in results]
    return [(x - min_val) / (max_val - min_val) for x in results]


def calculate_metrics(results):
    """
    Вычисляем простые метрики для анализа:
    - min
    - max
    - avg
    """
    if not results:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": min(results),
        "max": max(results),
        "avg": mean(results)
    }


def generate_report(metrics):
    """
    Формирует строку отчёта по метрикам.
    """
    return (f"Минимальная длина: {metrics['min']:.2f}\n"
            f"Максимальная длина: {metrics['max']:.2f}\n"
            f"Средняя длина: {metrics['avg']:.2f}")


# соберём pipeline
analysis_pipeline = compose(generate_report, calculate_metrics, normalize_data)


# ===========================
# 4. Сравнительный анализ для разного размера данных
# ===========================
@timing_decorator
def run_analysis(n_points, iterations=50):
    """
    Запускает ACO на n_points точках, собирает результаты и
    возвращает отчёт о метриках.
    """
    points = list(generate_points(n_points, bounds=(0, 200)))
    results = []

    # генерируем пути итерационно и сохраняем длины маршрутов
    aco_gen = ant_colony_optimization(points, iterations=iterations)
    for best_path, best_length in aco_gen:
        results.append(best_length)

    report = analysis_pipeline(results)
    print(f"\n Отчёт для {n_points} точек:\n{report}\n")
    return results


def main():
    """
    Тестируем алгоритм на наборах данных разного размера:
    200, 500, 1000 точек
    """
    print("Сравнительный анализ производительности ACO\n")
    for n in [200, 500, 1000]:
        run_analysis(n)


if __name__ == "__main__":
    main()

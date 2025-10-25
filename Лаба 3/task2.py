"""
task2_tsp.py

Генератор 3D-точек, генерация дорог (расстояний), и Ant Colony Optimization (ACO)
для приближённого решения задачи коммивояжёра в 3D.

Как запускать (пример):
    python task2_tsp.py

Нужен matplotlib (pip install matplotlib).
"""

import random
import math
import asyncio
from functools import partial
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (подключение 3D проекции)


# ----------------------------
# 1) Генератор случайных 3D-точек
# ----------------------------
def generate_points(n, bounds=(0, 100)):
    """
    Генератор n случайных точек в 3D.
    yields tuples (x, y, z)
    """
    low, high = bounds
    for _ in range(n):
        x = random.uniform(low, high)
        y = random.uniform(low, high)
        z = random.uniform(low, high)
        yield (x, y, z)


# ----------------------------
# 2) Генератор "дорог" / матриц расстояний
# ----------------------------
def euclidean_distance(a, b):
    """Евклидово расстояние между 3D-точками a и b."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def generate_roads(points, bidirectional_ratio=0.7):
    """
    Возвращает:
      - distances: NxN матрицу расстояний (symmetrical)
      - directed: NxN матрицу булевых значений (True если есть ребро i->j)
    Для простоты используем полную графовую матрицу расстояний, а directed управляет,
    какие связи считаем доступными (ориентированные или нет).
    """
    pts = list(points)
    n = len(pts)
    # матрица расстояний
    dist = [[0.0] * n for _ in range(n)]
    directed = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean_distance(pts[i], pts[j])
            dist[i][j] = d
            dist[j][i] = d

            # случайно делаем ребро двунаправленным или однонаправленным
            if random.random() < bidirectional_ratio:
                directed[i][j] = True
                directed[j][i] = True
            else:
                # однонаправленное: случайно выбираем направление
                if random.random() < 0.5:
                    directed[i][j] = True
                    directed[j][i] = False
                else:
                    directed[i][j] = False
                    directed[j][i] = True
    return dist, directed


# ----------------------------
# Вспомогательные функции для путей
# ----------------------------
def calculate_path_length(path, distances):
    """
    path: список индексов вершин [v0, v1, v2, ..., v0] (замкнутый путь)
    distances: матрица расстояний
    """
    if not path:
        return float('inf')
    total = 0.0
    # суммируем ребра по циклу
    for i in range(len(path) - 1):
        total += distances[path[i]][path[i+1]]
    return total


# ----------------------------
# 4) Компоненты ACO (муравьиный алгоритм)
# ----------------------------

def initialize_pheromones(n, initial=1.0):
    """Инициализация матрицы феромонов NxN значением initial."""
    return [[initial for _ in range(n)] for _ in range(n)]


def probability_transition(current, unvisited, pheromone, distances, alpha=1.0, beta=2.0):
    """
    Вычисление нормированных вероятностей перехода из текущей вершины в каждую
    из unvisited вершин (список индексов), основываясь на феромоне и видимости (1/distance).
    alpha — влияние феромона, beta — влияние расстояния (видимости).
    Возвращает список пар (node, prob).
    """
    numerators = []
    for j in unvisited:
        tau = pheromone[current][j] ** alpha
        eta = (1.0 / (distances[current][j] + 1e-9)) ** beta  # видимость
        numerators.append(tau * eta)

    total = sum(numerators)
    if total == 0:
        # равновероятное распределение
        prob = 1.0 / len(unvisited)
        return [(j, prob) for j in unvisited]

    probs = [num / total for num in numerators]
    return list(zip(unvisited, probs))


def select_next_by_prob(probs):
    """
    probs: список (node, prob), суммируют до 1.
    Выбираем случайно следующий узел.
    """
    r = random.random()
    cum = 0.0
    for node, p in probs:
        cum += p
        if r <= cum:
            return node
    # на случай ошибок
    return probs[-1][0]


def construct_path(pheromone, distances, start, directed=None, alpha=1.0, beta=2.0):
    """
    Построение одного пути муравья (замкнутого) начиная с вершины start.
    Использует вероятности переходов.
    Возвращает список индексов [v0, v1, ..., v0]
    """
    n = len(distances)
    visited = [start]
    current = start
    while len(visited) < n:
        # допустимые кандидаты — не посещённые и имеющие ребро (если directed указано)
        candidates = []
        for j in range(n):
            if j in visited:
                continue
            if directed is not None and not directed[current][j]:
                continue
            candidates.append(j)
        if not candidates:
            # если нет допустимых переходов (в ориентированном случае), то
            # пробуем допустить любые непосещённые
            candidates = [j for j in range(n) if j not in visited]

        probs = probability_transition(current, candidates, pheromone, distances, alpha, beta)
        nxt = select_next_by_prob(probs)
        visited.append(nxt)
        current = nxt

    # закрываем цикл — возвращаемся к старту (если есть ребро, иначе просто возвращаемся)
    visited.append(start)
    return visited


def evaporate_pheromones(pheromone, evaporation_rate):
    """Испарение: уменьшаем все феромоны на factor (1 - evaporation_rate)."""
    n = len(pheromone)
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1.0 - evaporation_rate)
    return pheromone


def deposit_pheromones(pheromone, path, amount):
    """
    Вносим феромон по ребрам пути. amount — добавляемая величина.
    path — список индексов [v0, v1, ..., v0]
    """
    for k in range(len(path) - 1):
        i = path[k]
        j = path[k+1]
        pheromone[i][j] += amount
        # если граф неориентирован, зеркально кладём тоже
        try:
            pheromone[j][i] += amount
        except Exception:
            pass
    return pheromone


def update_pheromones(pheromone, paths, distances, evaporation_rate=0.1, Q=100.0):
    """
    Обновление феромонов по всем путям (paths — итерируемый объект путей).
    Мы сначала испаряем, затем для каждого пути вносим Q / length.
    Здесь мы используем map/filter: фильтруем валидные пути, мапим на их длины.
    """
    n = len(pheromone)
    pheromone = evaporate_pheromones(pheromone, evaporation_rate)

    # фильтруем корректные пути (длина > 1)
    valid_paths = list(filter(lambda p: p and len(p) > 1, paths))

    # для каждого пути вычисляем длину и добавляем феромон
    for path in valid_paths:
        L = calculate_path_length(path, distances)
        if L <= 0:
            continue
        deposit_amount = Q / L
        deposit_pheromones(pheromone, path, deposit_amount)

    return pheromone


# ----------------------------
# 7) ant_colony_optimization (реализовано как генератор)
# ----------------------------
def ant_colony_optimization(points, iterations=100, num_ants=None,
                            alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0,
                            directed=None):
    """
    Генератор, позволяющий итерироваться по итерациям ACO.
    Каждый шаг возвращает (best_path, best_length).
    points: список 3d-точек
    directed: матрица булевых доступностей ребер или None
    """
    pts = list(points)
    n = len(pts)
    if num_ants is None:
        num_ants = n  # по одной муравью на точку по умолчанию

    distances, directed_matrix = generate_roads(pts) if directed is None else (None, directed)
    if distances is None:
        # если мы получили directed из аргумента, то пересчитаем distances:
        distances, _ = generate_roads(pts, bidirectional_ratio=1.0)

    # инициализация феромонов
    pheromone = initialize_pheromones(n, initial=1.0)

    # подготовка частичной функции для construct_path с фиксированными alpha/beta
    construct_with_params = partial(construct_path,
                                    pheromone, distances,
                                    alpha=alpha, beta=beta,
                                    directed=directed_matrix)

    # основной цикл итераций
    for it in range(iterations):
        # генератор путей — лениво создаём пути для каждого муравья
        paths_gen = (construct_with_params(random.randrange(n)) for _ in range(num_ants))

        # materialize paths (нужно для повторного прохода при обновлении феромонов)
        paths = list(paths_gen)

        # используем map для расчёта длин
        lengths = list(map(lambda p: calculate_path_length(p, distances), paths))

        # находим лучший путь текущей итерации
        best_idx = min(range(len(lengths)), key=lambda i: lengths[i])
        best_path = paths[best_idx]
        best_length = lengths[best_idx]

        # обновляем феромоны, давая преимущество лучшим путям
        pheromone = update_pheromones(pheromone, paths, distances,
                                      evaporation_rate=evaporation_rate, Q=Q)

        yield best_path, best_length


# ----------------------------
# 8) Асинхронная визуализация (matplotlib внутри run_in_executor)
# ----------------------------
async def visualize_optimization(algorithm_generator, points, interval=0.2):
    """
    Визуализация в отдельном потоке (через run_in_executor), чтобы
    не блокировать asyncio loop. algorithm_generator — генератор ACO.
    points — список 3D точек.
    """
    loop = asyncio.get_event_loop()

    # подготовим фигуру (в основном потоке через executor, т.к. GUI часто блокирует)
    def setup_plot():
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # scatter all points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        sc = ax.scatter(xs, ys, zs, c='blue', s=20)
        line, = ax.plot([], [], [], c='red', linewidth=2)
        text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
        plt.ion()
        plt.show()
        return fig, ax, line, text

    fig, ax, line, text = await loop.run_in_executor(None, setup_plot)

    # функция обновления, вызываемая в executor
    def update_plot_sync(path, length):
        if not path:
            return
        # path — список индексов; построим линии между точками
        xs = [points[idx][0] for idx in path]
        ys = [points[idx][1] for idx in path]
        zs = [points[idx][2] for idx in path]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        text.set_text(f"Length: {length:.2f}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    # итеративно обновляем визуализацию из генератора
    async for best_path, best_length in algorithm_generator:
        # обновляем в отдельном потоке
        await loop.run_in_executor(None, update_plot_sync, best_path, best_length)
        await asyncio.sleep(interval)


# ----------------------------
# Пример запуска: main()
# ----------------------------
def main_run_demo(n_points=100, iterations=200):
    # 1) Генерируем точки
    points = list(generate_points(n_points, bounds=(0, 200)))

    # 2) создаём генератор ACO
    aco_gen = ant_colony_optimization(points, iterations=iterations, alpha=1.0, beta=2.0, evaporation_rate=0.2, Q=100.0)

    # 3) В асинхронной среде запускаем визуализатор
    try:
        asyncio.run(visualize_optimization(aco_gen, points, interval=0.1))
    except KeyboardInterrupt:
        print("Визуализация остановлена пользователем.")


# если запущен как скрипт
if __name__ == "__main__":
    main_run_demo(n_points=80, iterations=100)


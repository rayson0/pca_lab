from normal import explained_variance_ratio
from hard import pca, reconstruction_error, pca
import math
import random
from typing import List, Tuple


def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
    eigenvalues: список собственных значений
    threshold: порог объяснённой дисперсии
    Выход: оптимальное число главных компонент k
    """
    n = len(eigenvalues)
    for i in range(1, n):
        shape = explained_variance_ratio(eigenvalues, i)
        if shape > threshold:
            return i
        
    return n


def column_mean(X, j):
    n = len(X)
    size = 0
    sum = 0
    for i in range(n):
        if not math.isnan(X[i][j]):
            sum += X[i][j]
            size += 1

    return sum / size


def handle_missing_values(X:'Matrix') ->'Matrix':
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    n = len(X)
    m = len(X[0])
    X_filled = X.copy()
    for i in range(n):
        for j in range(m):
            mean = column_mean(X, j)
            if math.isnan(X_filled[i][j]):
                X_filled[i][j] = mean

    return X_filled


def add_noise_and_compare(X:'Matrix', noise_level: float = 0.1) -> float:
    """
    Вход:
    X: матрица данных (n×m)
    noise_level: уровень шума (доля от стандартного отклонения)
    Выход: результаты PCA до и после добавления шума.
    В этом задании можете проявить творческие способности, поэтому выходные данные не
    типизированы.
    """
    mean = 0.0  # выберем среднее шума 0, чтобы не было закномерностей вверх или вниз
    X_noise = X.copy()
    for i in range(len(X)):
        for j in range(len(X[0])):
            noise = random.gauss(mean, noise_level)
            X_noise[i][j] = X[i][j] + noise

    result_before = pca(X)
    result_before_clear = handle_missing_values(result_before)

    result_after = pca(X_noise)
    result_after_clear = handle_missing_values(result_after)

    mse_error = reconstruction_error(result_before_clear, result_after_clear)

    return mse_error


def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
    """
    Вход:
    dataset_name: название датасета
    k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    with open(dataset_name) as f:
        lines = f.readlines()

    X, y = [], []
    for line in lines[1:]:  # пропускаем заголовок
        line = line.strip()
        if not line:
            continue

        row = line.split(",")

        y.append(float(row[-1]))
        X.append([float(val) for val in row[:-1]])

    result = pca(X, k)

    quality = calculate_geometry_preservation(X, result)

    return result, quality


def calculate_geometry_preservation(X_old, X_new):
    """
    Универсальная метрика: оценивает корреляцию расстояний ДО и ПОСЛЕ.
    Подходит для любых задач (регрессия, классификация, кластеризация).
    Выход: число от -1.0 (структура разрушена) до 1.0 (структура сохранена идеальна).
    """
    n = len(X_old)
    dim_old = len(X_old[0])
    dim_new = len(X_new[0])

    distances_old = []
    distances_new = []

    # 1. Считаем все попарные расстояния ДО и ПОСЛЕ
    for i in range(n):
        for j in range(i + 1, n):
            # Расстояние в исходном пространстве
            sum_old = 0.0
            for d in range(dim_old):
                sum_old += (X_old[i][d] - X_old[j][d]) ** 2
            distances_old.append(math.sqrt(sum_old))

            # Расстояние в пространстве PCA
            sum_new = 0.0
            for d in range(dim_new):
                sum_new += (X_new[i][d] - X_new[j][d]) ** 2
            distances_new.append(math.sqrt(sum_new))

    # 2. Считаем корреляцию Пирсона между списками расстояний вручную
    m = len(distances_old)
    if m == 0:
        return 0.0

    mean_old = sum(distances_old) / m
    mean_new = sum(distances_new) / m

    covariance = 0.0
    var_old = 0.0
    var_new = 0.0

    for i in range(m):
        diff_old = distances_old[i] - mean_old
        diff_new = distances_new[i] - mean_new
        covariance += diff_old * diff_new
        var_old += diff_old**2
        var_new += diff_new**2

    std_prod = math.sqrt(var_old * var_new)
    if std_prod == 0:
        return 0.0

    return covariance / std_prod


# https://www.kaggle.com/datasets/abrambeyer/openintro-possum

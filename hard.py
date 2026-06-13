from easy import center_data, covariance_matrix
from normal import find_eigenvalues, find_eigenvectors, explained_variance_ratio
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple

def pca(X:'Matrix', k: int = 0) -> Tuple['Matrix', float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """
    from expert import auto_select_k

    n = len(X)
    m = len(X[0])

    X_centered = center_data(X)
    C = covariance_matrix(X_centered)

    all_eigenvalues = find_eigenvalues(C)
    k_optimization = auto_select_k(all_eigenvalues)
    eigenvalues = all_eigenvalues[:k_optimization]
    eigenvectors = find_eigenvectors(C, eigenvalues)

    X_pca = [[0] * k_optimization for _ in range(n)]
    for i in range(n):
        for j in range(k_optimization):
            X_pca[i][j] = sum(X_centered[i][idx] * eigenvectors[j][idx] for idx in range(m))

    share = explained_variance_ratio(eigenvalues, k_optimization)

    return X_pca, share


def plot_pca_projection(X_proj:'Matrix') -> Figure:
    """
    Вход: проекция данных X_proj (n×2)
    Выход: объект Figure из Matplotlib
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for x, y in X_proj:
        ax.scatter(x, y)

    ax.set_xlabel("Первая главная компонента", fontsize=12, fontweight="bold")
    ax.set_ylabel("Вторая главная компонента", fontsize=12, fontweight="bold")
    ax.set_title(
        "Проекция данных на первые две главные компоненты", fontsize=14, pad=15
    )

    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    return fig


def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    """
    Вход:
    X_orig: исходные данные (n×m)
    X_recon: восстановленные данные (n×m)
    Выход: среднеквадратическая ошибка MSE
    """
    n = len(X_orig)
    m = len(X_orig[0])

    total_squared_error = 0.0
    for i in range(n):
        for j in range(m):
            diff = X_orig[i][j] - X_recon[i][j]
            total_squared_error += diff * diff

    total_elements = n * m

    return total_squared_error / total_elements

import math
from typing import List
from easy import gauss_solver


def find_eigenvalues(C, tol: float = 1e-6) -> List[float]:
    n = len(C)
    if n == 0:
        return []
    if n == 1:
        return [C[0][0]]

    A = [row[:] for row in C]

    # Отражения Хаусхолдера (приведение к трехдиагональному виду A = Q^T * C * Q)
    for k in range(n - 2):
        x = [A[i][k] for i in range(k + 1, n)]
        norm_x = math.sqrt(sum(val**2 for val in x))

        if norm_x == 0:
            continue

        s = norm_x if x[0] <= 0 else -norm_x

        u = [0.0] * len(x)
        u[0] = x[0] - s
        for i in range(1, len(x)):
            u[i] = x[i]

        norm_u = math.sqrt(sum(val**2 for val in u))
        if norm_u == 0:
            continue

        v = [val / norm_u for val in u]

        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                H[i][j] -= 2.0 * v[i - (k + 1)] * v[j - (k + 1)]

        # HA = H * A
        HA = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                HA[i][j] = sum(H[i][m] * A[m][j] for m in range(n))

        # A = HA * H
        for i in range(n):
            for j in range(n):
                A[i][j] = sum(HA[i][m] * H[m][j] for m in range(n))

    diag = [A[i][i] for i in range(n)]
    subdiag = [A[i + 1][i] for i in range(n - 1)]

    return find_all_eigenvalues(diag, subdiag, tol)


def count_sign_changes(diag, subdiag, lam):
    """Корректный подсчет смен знака в последовательности Штурма"""
    n = len(diag)
    count = 0

    p_prev = 1.0
    p_curr = diag[0] - lam
    if p_curr == 0.0:
        p_curr = 1e-14

    if p_prev * p_curr < 0:
        count += 1

    for k in range(1, n):
        alpha = diag[k] - lam
        beta = subdiag[k - 1]

        p_next = alpha * p_curr - (beta**2) * p_prev

        if p_next == 0.0:
            p_next = 1e-14

        if p_curr * p_next < 0:
            count += 1

        p_prev = p_curr
        p_curr = p_next

    return count


def find_all_eigenvalues(diag, subdiag, tol=1e-6):
    n = len(diag)
    a_global = float("inf")
    b_global = float("-inf")

    # Теорема Гершгорина для определения границ спектра
    for i in range(n):
        r = 0.0
        if i > 0:
            r += abs(subdiag[i - 1])
        if i < n - 1:
            r += abs(subdiag[i])

        row_min = diag[i] - r
        row_max = diag[i] + r

        if row_min < a_global:
            a_global = row_min
        if row_max > b_global:
            b_global = row_max

    eigenvalues = []
    queue = [(a_global, b_global, 0, n)]

    while queue:
        left, right, n_left, n_right = queue.pop(0)

        if n_right - n_left == 0:
            continue

        if (right - left) < tol:
            for _ in range(n_right - n_left):
                eigenvalues.append((left + right) / 2.0)
            continue

        mid = (left + right) / 2.0
        n_mid = count_sign_changes(diag, subdiag, mid)

        if n_mid > n_left:
            queue.append((left, mid, n_left, n_mid))

        if n_right > n_mid:
            queue.append((mid, right, n_mid, n_right))

    return sorted(eigenvalues, reverse=True)


def find_eigenvectors(C, eigenvalues):
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    n = len(C)
    if n == 1:
        return 0

    solves = []
    C_new = [row[:] for row in C]
    for lam in eigenvalues:
        for i in range(n):
            for j in range(n):
                C_new[i][j] = C[i][j] - lam if i == j else C[i][j]

        solves.extend(gauss_solver(C_new, [0] * n))

    return solves


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент
    Выход: доля объяснённой дисперсии
    """
    numerator = 0
    denominator = 0
    for i in range(len(eigenvalues)):
        if i <= k:
            numerator += eigenvalues[i]
        denominator += eigenvalues[i]
    
    return numerator / denominator

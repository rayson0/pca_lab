from typing import List

def gauss_solver(A, b) -> List['matrix']:
    """
    Вход:
    A: матрица коэффициентов (n×n). Используется класс A_ext из предыдущей
    лабораторной работы→
    b: вектор правых частей (n×1)
    Выход:
    Raises:
    list[A_ext]: список базисных векторов решения системы
    ValueError: если система несовместна
    """
    A_ext = [A[i] + [b[i]] for i in range(len(A))]
    for col in range(len(A_ext)):
        main_num = A_ext[col][col]
        if main_num == 0:
            if col == len(A_ext) - 1 and A_ext[col][col + 1] != 0:
                raise ValueError("Система несовместна")
            elif col == len(A_ext) - 1 and A_ext[col][col + 1] == 0:
                del A_ext[col]
            else:
                row_new = col + 1
                while A_ext[row_new][col] != 0 and row_new < len(A_ext) - 1:
                    row_new += 1

                A_ext[row_new], A_ext[col] = A_ext[col], A_ext[row_new]
                main_num = A_ext[col][col]

        for row in range(col + 1, len(A_ext)):            
            num_row = A_ext[row][col]
            if num_row != 0:
                A_ext[row] = [A_ext[row][i] + (-num_row / main_num) * A_ext[col][i] for i in range(len(A_ext) + 1)]

    row = len(A_ext) - 1
    while all([item == 0 for item in A_ext[row][:-1]]) and row >= 0:
        if A_ext[row][-1] != 0:
            raise ValueError('Система несовместна')

        del A_ext[row]
        row -= 1

    m = len(A_ext)  # кол-во независимых строк
    n = len(A)  # кол-во переменных
    cols_ext = len(A) + 1  # кол-во столбцов в расширенной матрице
    EPS = 1e-5

    basis_variables = {}
    pivot_variables = set()

    for i in range(m):
        for j in range(cols_ext):
            if abs(A_ext[i][j]) > EPS:
                pivot_variables.add(j)
                basis_variables[i] = j
                break

    free_cols = set(range(n)) - pivot_variables
    # Тривиальное решение однородной СЛАУ
    if (not free_cols) and all(item == 0 for item in b):
        return [[0] * n]
    # Единственное решение неоднородной СЛАУ
    elif (not free_cols) and any(item != 0 for item in b):
        solve = [0] * n
        for i in range(m - 1, -1, -1):
            current_sum = A_ext[i][cols_ext - 1]

            for j in range(i + 1, n):
                current_sum -= A_ext[i][j] * solve[j]

            solve[i] = current_sum / A_ext[i][i]

        return [solve]

    basis_solves = [[0] * n] if all(item == 0 for item in b) else []
    for var_idx in free_cols:
        solve = [0] * n
        solve[var_idx] = 1
        for i in range(m - 1, -1, -1):
            current_sum = A_ext[i][cols_ext - 1]
            for j in range(i + 1, n):
                current_sum -= A_ext[i][j] * solve[j]

            solve[i] = current_sum / A_ext[i][i]

        basis_solves.append(solve)

    return basis_solves


def center_data(X):
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    if not X:
        return 0
    
    sum = 0
    n = len(X)
    m = len(X[0])
    for row in range(n):
        for col in range(m):
            sum += X[row][col]

    

    X_new = []
    for row in range(n):
        row_new = [X[row][col] - sum / (n*m) for col in range(m)]
        X_new.append(row_new)

    return X_new


def covariance_matrix(X_centered):
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    if not X_centered:
        return 0
    
    n = len(X_centered)
    m = len(X_centered[0])
    C = []
    for row in range(m):
        for col in range(m):
            C[row][col] = 1 / (n-1) * sum([X_centered[i][row] * X_centered[i][col] for i in range(m)])

    return C
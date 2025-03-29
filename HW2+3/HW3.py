import numpy as np

def cholesky_decomposition(A):
    """
    Выполняет разложение Холецкого матрицы A.
    Предполагается, что A симметричная и положительно определённая.
    
    Возвращает нижнетреугольную матрицу L такую, что A = L * L.T
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_sq = np.sum(L[i, :j] ** 2)
                L[i, j] = np.sqrt(A[i, i] - sum_sq)
            else:
                sum_mul = np.sum(L[i, :j] * L[j, :j])
                L[i, j] = (A[i, j] - sum_mul) / L[j, j]
    return L

def forward_substitution(L, b):
    """
    Решает систему L y = b методом прямой подстановки,
    где L - нижнетреугольная матрица.
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(U, y):
    """
    Решает систему U x = y методом обратной подстановки,
    где U - верхнетреугольная матрица.
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_cholesky(A, b):
    """
    Решает СЛАУ A x = b методом квадратного корня (разложение Холецкого).
    
    1. Находит разложение A = L * L.T.
    2. Решает L y = b (прямая подстановка).
    3. Решает L.T x = y (обратная подстановка).
    """
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x

# Пример использования
if __name__ == '__main__':
    # Пример симметричной положительно определённой матрицы A и вектора b
    A = np.array([[5.1, 7, 6, 5],
                  [7, 10.1, 8, 7],
                  [6, 8, 10.1, 9],
                  [5, 7, 9, 10.1]], dtype=float)
    b = np.array([23, 32, 33, 31], dtype=float)
    
    x = solve_cholesky(A, b)
    print("Решение x:", x)

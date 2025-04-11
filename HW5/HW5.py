import numpy as np

# Заданная матрица A и вектор b
A = np.array([
    [18.22,  1.44, -1.72,  1.91],
    [ 1.44, 17.33,  1.11, -1.82],
    [-1.72,  1.11, 17.24,  1.42],
    [ 1.91, -1.82,  1.42, 18.55]
])
b = np.array([7.53, 6.06, 8.05, 8.06])


def gauss_seidel_fixed(A, b, iterations):
    """
    Решает систему A x = b методом Гаусса–Зейделя, выполняя ровно iterations итераций.
    
    Параметры:
        A : np.ndarray, квадратная матрица системы
        b : np.ndarray, вектор правой части
        iterations : int, число итераций (k)
    
    Возвращает:
        x : np.ndarray, приближённое решение после заданного числа итераций
    """
    n = len(b)
    x = np.zeros_like(b, dtype=float)  # Начальное приближение
    
    for _ in range(iterations):
        for i in range(n):
            # Сумма по уже обновленным значениям x[0:i]
            s1 = sum(A[i, j] * x[j] for j in range(i))
            # Сумма по старым значениям x[i+1:n]
            s2 = sum(A[i, j] * x[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
    return x


def chebyshev_iteration_fixed(A, b, iterations):
    """
    Итерационный метод с чебышёвским набором параметров (с диагональным предобуславливанием),
    выполняющий ровно iterations итераций.
    
    Параметры:
        A : np.ndarray, матрица системы
        b : np.ndarray, вектор правой части
        iterations : int, число итераций (узлов) чебышёвских параметров
    
    Возвращает:
        x : np.ndarray, приближённое решение после заданного числа итераций
    """
    n = len(b)
    D = np.diag(A)
    
    # Вычисляем собственные значения матрицы A для оценки спектра.
    eigvals = np.linalg.eigvals(A)
    lambda_min = np.min(eigvals)
    lambda_max = np.max(eigvals)
    
    # Начальное приближение
    x = np.zeros_like(b, dtype=float)
    
    # Параметр xi для чебышёвских коэффициентов
    xi = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    
    for k in range(1, iterations+1):
        theta_k = np.pi * (2*k - 1) / (2 * iterations)
        alpha_k = (2.0 / (lambda_max + lambda_min)) / (1.0 - xi * np.cos(theta_k))
        
        # Итерационный шаг (аналог метода Якоби с релаксацией)
        r = A @ x - b  # вычисляем невязку
        x = x - alpha_k * (r / D)
    return x


if __name__ == "__main__":
    # Для k = 7
    k1 = 7
    x_gs_7 = gauss_seidel_fixed(A, b, k1)
    residual_gs_7 = np.linalg.norm(A @ x_gs_7 - b)
    
    x_cheb_7 = chebyshev_iteration_fixed(A, b, k1)
    residual_cheb_7 = np.linalg.norm(A @ x_cheb_7 - b)
    
    print("Результаты после k = 7 итераций:")
    print("Метод Гаусса–Зейделя:")
    print("x =", x_gs_7)
    print("Невязка =", residual_gs_7)
    print("\nМетод чебышёвской итерации:")
    print("x =", x_cheb_7)
    print("Невязка =", residual_cheb_7)
    
    print("\n" + "-"*50 + "\n")
    
    # Для k = 15
    k2 = 15
    x_gs_15 = gauss_seidel_fixed(A, b, k2)
    residual_gs_15 = np.linalg.norm(A @ x_gs_15 - b)
    
    x_cheb_15 = chebyshev_iteration_fixed(A, b, k2)
    residual_cheb_15 = np.linalg.norm(A @ x_cheb_15 - b)
    
    print("Результаты после k = 15 итераций:")
    print("Метод Гаусса–Зейделя:")
    print("x =", x_gs_15)
    print("Невязка =", residual_gs_15)
    print("\nМетод чебышёвской итерации:")
    print("x =", x_cheb_15)
    print("Невязка =", residual_cheb_15)

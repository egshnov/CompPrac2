import numpy as np

def jacobi_iteration(A, b, max_iter=100, tol=1e-5):
    """
    Решение системы A x = b методом простой итерации (Якоби).
    
    Параметры:
    ----------
    A : ndarray
        Квадратная матрица системы (n x n).
    b : ndarray
        Вектор правой части (n, ).
    max_iter : int
        Максимальное число итераций.
    tol : float
        Требуемая точность (критерий остановки по норме разности двух последовательных приближений).
        
    Возвращает:
    -----------
    x : ndarray
        Итоговое приближение решения.
    k : int
        Фактическое число итераций до выхода.
    history : list
        История приближений на каждом шаге.
    """
    n = len(b)
    
    # Разделяем A на диагональную и остальные части
    D = np.diag(np.diag(A))         # Диагональная матрица
    R = A - D                       # Остальная часть
    
    # Считаем H и g
    D_inv = np.diag(1 / np.diag(D)) # D^{-1}
    H = -D_inv @ R
    g = D_inv @ b
    
    # Начальное приближение (нулевой вектор)
    x_old = np.zeros(n)
    
    history = [x_old]
    
    for k in range(1, max_iter+1):
        # Итерация: x^{(k+1)} = H x^{(k)} + g
        x_new = H @ x_old + g
        
        # Сохраняем в историю
        history.append(x_new)
        
        # Проверяем, достаточно ли мы приблизились
        if np.linalg.norm(x_new - x_old, ord=np.inf) < tol:
            return x_new, k, history
        
        x_old = x_new
    
    return x_old, max_iter, history


def apriori_estimate(H, x_init, x1, eps):
    """
    Априорная оценка числа итераций, чтобы обеспечить точность eps.
    Используем оценку вида:
        ||x^(k) - x*|| <= (||H||^k / (1 - ||H||)) * ||x^(1) - x^(0)||,
    если норма ||H|| < 1.
    
    Возвращаем оценку k, при котором погрешность будет не хуже eps.
    """
    # Возьмём, к примеру, норму ||.||_inf (максимум по строкам)
    normH = np.linalg.norm(H, ord=np.inf)
    if normH >= 1:
        # Теория гарантированной сходимости для ||H|| < 1
        raise ValueError("Норма H >= 1, априорная оценка в таком виде неприменима")
    
    # ||x^(1) - x^(0)||_inf
    e0 = np.linalg.norm(x1 - x_init, ord=np.inf)
    
    # Нужно найти минимальное k, такое что:
    #   (normH^k / (1 - normH)) * e0 < eps
    # Перепишем неравенство:
    #   normH^k < eps * (1 - normH) / e0
    # Применим логарифм:
    #   k * ln(normH) < ln( eps*(1-normH)/e0 )
    #   k > ln( eps*(1-normH)/e0 ) / ln(normH)   (учитывая, что 0 < normH < 1, ln(normH)<0)
    
    right_side = np.log(eps * (1 - normH) / e0) / np.log(normH)
    # Поскольку логарифм normH < 0, деление переворачивает знак неравенства.
    # Но нам нужен ceil от модуля этого выражения.
    # Аккуратно берём int, добавляя 1.
    k_est = int(np.ceil(right_side))
    return max(k_est, 1)


def solve_via_jacobi():
    # Матрица A (из предыдущего шага: A = D + 12*C)
    A = np.array([
        [18.22,  1.44, -1.72,  1.91],
        [ 1.44, 17.33,  1.11, -1.82],
        [-1.72,  1.11, 17.24,  1.42],
        [ 1.91, -1.82,  1.42, 18.55]
    ], dtype=float)
    
    # Вектор b
    b = np.array([7.53, 6.06, 8.05, 8.06], dtype=float)
    
    # Считаем решение методом простой итерации:
    # 1) Сразу сделаем 15 итераций
    x_15, _, history = jacobi_iteration(A, b, max_iter=15, tol=1e-15)
    
    print("Приближение после 15 итераций (x^(15)):\n", x_15)
    
    # 2) Априорная оценка для k=15 (если нужно сравнить с погрешностью и т.д.)
    #    Для этого возьмём H, g изнутри (снова вычислим):
    n = len(b)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1 / np.diag(D))
    H = -D_inv @ R
    g = D_inv @ b
    
    x0 = np.zeros(n)      # начальное приближение
    x1 = H @ x0 + g       # первое приближение
    
    # Попробуем оценить погрешность на 15-м шаге
    normH = np.linalg.norm(H, ord=np.inf)
    if normH < 1:
        # Оценка ||x^(15) - x*|| <= (||H||^15 / (1 - ||H||)) * ||x^(1) - x^(0)||
        e0 = np.linalg.norm(x1 - x0, ord=np.inf)
        apri_15 = (normH**15 / (1 - normH)) * e0
        print(f"Априорная оценка погрешности на 15-й итерации: {apri_15: .3e}")
    else:
        print("Норма H >= 1, метод может не сходиться.")
        apri_15 = None
    
    # 3) Найдём k, при котором априорная оценка даст точность eps = 1e-5
    eps = 1e-5
    if normH < 1:
        k_needed = apriori_estimate(H, x0, x1, eps)
        print(f"По априорной оценке, для точности {eps} нужно примерно k = {k_needed} итераций.")
        
        # 4) Запустим итерации до k_needed
        x_k, actual_k, history_k = jacobi_iteration(A, b, max_iter=k_needed, tol=1e-15)
        
        # 5) Апостериорная оценка: можно сравнить x_k с x_{k-1} или
        #    найти точное решение и посчитать норму разности.
        #    Точное решение (для проверки) найдём через np.linalg.solve
        x_true = np.linalg.solve(A, b)
        post_est = np.linalg.norm(x_true - x_k, ord=np.inf)
        
        print(f"Фактическое решение за {k_needed} итераций: {x_k}")
        print(f"Точное решение через numpy.linalg.solve:   {x_true}")
        print(f"Апостериорная невязка ||x^(k) - x*||_∞ = {post_est: .3e}")
    else:
        print("Априорная оценка не применима, так как ||H|| >= 1.")


if __name__ == "__main__":
    solve_via_jacobi()

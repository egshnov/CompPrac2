import numpy as np

def compact_gauss(A):
    """
    Выполняет in-place LU-разложение (компактную схему Гаусса) матрицы A без перестановок.
    После разложения:
      - Элементы на диагонали и выше содержат матрицу U.
      - Элементы ниже диагонали содержат множители для L (на диагонали L предполагаются единицы).
    
    Возвращает разложенную матрицу (копию исходной матрицы A).
    """
    A = A.copy().astype(float)
    n = A.shape[0]
    for k in range(n):
        # Делим элементы столбца ниже диагонали на ведущий элемент
        for i in range(k+1, n):
            A[i, k] /= A[k, k]
        # Обновляем оставшуюся часть матрицы
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[i, j] -= A[i, k] * A[k, j]
    return A

def determinant_from_lu(LU):
    """
    Вычисляет определитель матрицы A, если дана компактная LU-разложенная матрица.
    Так как A = L*U, а diag(L)=1, то det(A) = det(U) = произведение диагональных элементов.
    """
    return np.prod(np.diag(LU))

def forward_substitution(LU, b):
    """
    Решает систему L y = b методом прямой подстановки.
    Здесь L — нижнетреугольная матрица с единичной диагональю, элементы L[i,j] для i>j хранятся в LU.
    """
    n = LU.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= LU[i, j] * y[j]
        y[i] = s  # т.к. L[i,i] = 1
    return y

def backward_substitution(LU, y):
    """
    Решает систему U x = y методом обратной подстановки.
    U — верхнетреугольная матрица, элементы которой хранятся в LU (диагональ и выше).
    """
    n = LU.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= LU[i, j] * x[j]
        x[i] = s / LU[i, i]
    return x

def solve_lu(LU, b):
    """
    Решает систему A x = b, если дана компактная LU-разложенная матрица LU.
    Сначала решается L y = b, затем U x = y.
    """
    y = forward_substitution(LU, b)
    x = backward_substitution(LU, y)
    return x

def inverse_from_lu(LU):
    """
    Вычисляет обратную матрицу A⁻¹ по LU-разложению.
    Для каждого столбца единичной матрицы решается система A x = e.
    """
    n = LU.shape[0]
    A_inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        A_inv[:, i] = solve_lu(LU, e)
    return A_inv

def condition_number(A, norm_type=1):
    """
    Вычисляет число обусловленности матрицы A в выбранной норме.
    Число обусловленности определяется как:
        κ(A) = ||A|| * ||A⁻¹||.
    Здесь обратная матрица A⁻¹ вычисляется через компактное LU-разложение.
    
    Если определитель A равен 0, возвращается бесконечность.
    """
    LU = compact_gauss(A)
    detA = determinant_from_lu(LU)
    if detA == 0:
        return np.inf, detA
    A_inv = inverse_from_lu(LU)
    norm_A = np.linalg.norm(A, ord=norm_type)
    norm_A_inv = np.linalg.norm(A_inv, ord=norm_type)
    cond = norm_A * norm_A_inv
    return cond, detA

# Пример использования
if __name__ == '__main__':
    # Пример матрицы A
    A = np.array([[2.2, -4.5, -2.0],
                  [3.0, 2.6,   4.3],
                  [-6, 3.5,    2.6]], dtype=float)
    
    # Получаем компактное LU-разложение
    LU = compact_gauss(A)
    
    # Вычисляем определитель
    detA = determinant_from_lu(LU)
    print("Определитель A =", detA)
    
    # Вычисляем обратную матрицу
    A_inv = inverse_from_lu(LU)
    print("Обратная матрица A:")
    print(A_inv)
    
    # Вычисляем число обусловленности в 1-норме
    cond, _ = condition_number(A, norm_type=1)
    print("Число обусловленности A (1-норма) =", cond)

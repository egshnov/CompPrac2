import numpy as np

# ======================================================
# 1. Задаём матрицу A = D + y*C (примерные данные)
# ======================================================

D = np.array([
    [1.0, -3, 5, 7],
    [-3, 9, 11, 13],
    [5, 11, 15, -17],
    [7, 13, -17, 19]
])

# Пример матрицы C (замените на свои данные)
C = np.array([
    [0.5, 0, 1, 1],
    [0, 0.5, 0, 0],
    [1, 0, 0.5, 0],
    [1, 0, 0, 0.5]
])
# Параметр y = 1
y = 1

# Формируем матрицу A
A = D + y*C

# ======================================================
# 2. Метод вращений Якоби (Jacobi rotation method)
#    для поиска собственных значений и векторов
# ======================================================

def jacobi_rotation_method(A, tol=1e-10, max_iter=1000):
    """
    Находит все собственные значения и собственные векторы
    симметричной матрицы A методом вращений Якоби.
    
    Параметры:
        A        : np.ndarray (n x n), предполагается симметричной
        tol      : float, точность для максимального по модулю внедиагонального элемента
        max_iter : int, максимальное число итераций

    Возвращает:
        eigenvalues  : np.ndarray, массив собственных значений (размер n)
        eigenvectors : np.ndarray, матрица (n x n), столбцы - собственные векторы
    """
    # Копируем A, чтобы не менять исходную
    A = A.copy()
    n = A.shape[0]
    
    # Матрица для накопления собственных векторов
    # Начинаем с единичной матрицы (V = I)
    V = np.eye(n)
    
    for _ in range(max_iter):
        # 1. Находим максимальный по модулю внедиагональный элемент
        off_diag_max = 0.0
        p, q = 0, 1
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i,j]) > abs(off_diag_max):
                    off_diag_max = A[i,j]
                    p, q = i, j
        
        # Если он меньше tol, значит матрица почти диагональна
        if abs(off_diag_max) < tol:
            break
        
        # 2. Вычисляем угол поворота
        # Для симметричной матрицы A[p,p], A[q,q], A[p,q]
        # Формулы классической схемы Якоби
        alpha = (A[q,q] - A[p,p]) / (2.0 * A[p,q])
        
        # t = sign(alpha) / (|alpha| + sqrt(1 + alpha^2))
        # phi = arctan(t)
        t = np.sign(alpha) / (abs(alpha) + np.sqrt(1.0 + alpha**2))
        c = 1.0 / np.sqrt(1.0 + t**2)
        s = t * c
        
        # 3. Обновляем элементы матрицы A
        # Сохраним A[p,p], A[q,q] и A[p,q]
        app = A[p,p]
        aqq = A[q,q]
        apq = A[p,q]
        
        # Новые диагональные элементы
        A[p,p] = app - t * apq
        A[q,q] = aqq + t * apq
        A[p,q] = 0.0
        A[q,p] = 0.0  # симметрия
        
        # Обновляем остальные элементы
        for i in range(n):
            if i != p and i != q:
                aip = A[i,p]
                aiq = A[i,q]
                A[i,p] = c*aip - s*aiq
                A[p,i] = A[i,p]  # симметричность
                
                A[i,q] = c*aiq + s*aip
                A[q,i] = A[i,q]
        
        # 4. Обновляем матрицу собственных векторов V
        for i in range(n):
            vip = V[i,p]
            viq = V[i,q]
            V[i,p] = c*vip - s*viq
            V[i,q] = s*vip + c*viq
    
    # После завершения итераций, диагональ A - это собственные значения
    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors

# ======================================================
# 3. Теорема Гершгорина: определяем круги (диски) Гершгорина
# ======================================================

def gershgorin_disks(A):
    """
    Для каждой строки i матрицы A возвращает (center_i, radius_i),
    где center_i = A[i,i], а radius_i = сумма модулей внедиагональных
    элементов в той же строке i.
    """
    n = A.shape[0]
    disks = []
    for i in range(n):
        center = A[i,i]
        radius = np.sum(np.abs(A[i,:])) - abs(A[i,i])  # сумма по всей строке, минус диагональ
        disks.append((center, radius))
    return disks

# ======================================================
# 4. Основной блок: собираем всё вместе и выводим результаты
# ======================================================
if __name__ == "__main__":
    print("Матрица A = D + y*C при y=1:")
    print(A, "\n")
    
    # 4.1 Находим собственные значения и векторы методом вращений Якоби
    eigenvals, eigenvecs = jacobi_rotation_method(A, tol=1e-12, max_iter=1000)
    
    # Сортируем по возрастанию (для удобства вывода)
    # и сортируем соответствующие столбцы векторов
    idx_sorted = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[idx_sorted]
    eigenvecs_sorted = eigenvecs[:, idx_sorted]
    
    print("Собственные значения (метод Якоби):")
    for i, val in enumerate(eigenvals_sorted):
        print(f"  lambda_{i+1} = {val}")
    print()
    
    print("Собственные векторы (столбцы, соответствующие lambda_i):")
    print(eigenvecs_sorted, "\n")
    
    # 4.2 Теорема Гершгорина: построим гершгориновы круги
    disks = gershgorin_disks(A)
    print("Гершгориновы круги (center, radius) по строкам матрицы A:")
    for i, (center, radius) in enumerate(disks):
        left = center - radius
        right = center + radius
        print(f"  Строка {i}: центр = {center:.4f}, радиус = {radius:.4f}  =>  [{left:.4f}, {right:.4f}]")
    print()
    
    print("Проверка, что найденные собственные значения попадают в объединение этих кругов:")
    for val in eigenvals_sorted:
        # Проверяем, лежит ли val хотя бы в одном из отрезков [center - r, center + r]
        in_any_disk = any((val >= (c - r) and val <= (c + r)) for (c, r) in disks)
        print(f"  lambda = {val:.6f}   =>  {'входит' if in_any_disk else 'НЕ входит'} в гершгориновы круги")

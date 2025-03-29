def gaussian_elimination_full_pivoting(A, b, tol=1e-12):
    """
    Решает систему линейных уравнений A*x = b методом Гаусса с полным пивотированием.
    
    Параметры:
        A : список списков (матрица коэффициентов, размером n x n)
        b : список свободных членов
        tol : порог для сравнения с нулём (по умолчанию 1e-12)
    
    Возвращает:
        x : список решений, расположенных в исходном порядке переменных
    """
    
    n = len(A)
    # Формируем расширенную матрицу [A | b]
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]
    # Вектор для отслеживания перестановок столбцов (исходный порядок переменных)
    col_perm = list(range(n))
    
    # Прямой ход с полным пивотированием
    for i in range(n):
        # Ищем максимальный по модулю элемент в подматрице M[i:n][i:n]
        max_val = 0
        pivot_row = i
        pivot_col = i
        for row in range(i, n):
            for col in range(i, n):
                if abs(M[row][col]) > abs(max_val):
                    max_val = M[row][col]
                    pivot_row = row
                    pivot_col = col
        # Если максимальный элемент меньше tol, считаем его нулевым
        if abs(max_val) < tol:
            # Если свободный член в этой строке не равен нулю, система несовместна
            if abs(M[i][-1]) > tol:
                raise ValueError("Система несовместна (обнаружена строка вида [0 ... 0 | b] с b ≠ 0)")
            else:
                # Вырожденная система (бесконечное число решений или избыточные уравнения)
                continue

        # Меняем местами строки: текущая i-я строка и строка с найденным пивотом
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]

        # Меняем местами столбцы: текущий i-й столбец и столбец с найденным пивотом
        # При этом обновляем вектор перестановок
        if pivot_col != i:
            for row in range(n):
                M[row][i], M[row][pivot_col] = M[row][pivot_col], M[row][i]
            col_perm[i], col_perm[pivot_col] = col_perm[pivot_col], col_perm[i]
        
        # Нормализация текущей строки: делим строку на ведущий элемент
        pivot = M[i][i]
        M[i] = [elem / pivot for elem in M[i]]
        
        # Исключаем переменную из строк ниже текущей
        for row in range(i + 1, n):
            factor = M[row][i]
            M[row] = [elem_row - factor * elem_i for elem_row, elem_i in zip(M[row], M[i])]
    
    # Обратный ход: обратная подстановка
    x_temp = [0] * n
    for i in range(n - 1, -1, -1):
        x_temp[i] = M[i][-1] - sum(M[i][j] * x_temp[j] for j in range(i + 1, n))
    
    # Восстанавливаем порядок решений согласно исходному расположению переменных
    x = [0] * n
    for i in range(n):
        x[col_perm[i]] = x_temp[i]
    
    return x


if __name__ == "__main__":
    A = [
        [2.2, -4.5, -2.0],
        [3.0, 2.6,   4.3],
        [-6, 3.5,    2.6]
    ]
    
    b = [19.07, 3.21, -18.25]
    
    try:
        solution = gaussian_elimination_full_pivoting(A, b)
        print("Решение системы:", solution)
    except ValueError as e:
        print("Ошибка:", e)

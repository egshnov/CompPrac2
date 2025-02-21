# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import scipy

a = 0.0 
b = 1.0
n = 10 
h = (b - a) / n 
x_nodes = np.linspace(a, b, n+1)  # Узлы сетки x_i


y = np.zeros(n+1)  # Массив для численного решения
c = np.zeros(n+1)  #(c_i)
d = np.zeros(n+1)  #(d_i)


# (y_{i+1} - 2y_i + y_{i-1})/h² + 5.5y_i = -x_i


# y_{i-1} - (2 - 5.5h²)y_i + y_{i+1} = -h² x_i

# Прямой ход метода прогонки 
for i in range(1, n):  
    a_i = 1.0          # Коэффициент при y_{i-1}
    b_i = -(2 - 5.5 * h**2)  # Коэффициент при y_i
    f_i = -h**2 * x_nodes[i]  # Правая часть уравнения
    
    # Учет левого краевого условия y(0) = 0 при i=1
    if i == 1:
        c[i] = 1.0 / b_i  # Начальный коэффициент c_1
        d[i] = (f_i - a_i * 0) / b_i  # Начальный коэффициент d_1 (y_0=0)
    else:
        # Рекуррентные формулы для c_i и d_i
        c[i] = 1.0 / (b_i - a_i * c[i-1])
        d[i] = (f_i - a_i * d[i-1]) / (b_i - a_i * c[i-1])

# Обратный ход метода прогонки
y[-1] = 0.0  # Правое граничное условие y(1) = 0
for i in range(n-1, 0, -1):  # От предпоследнего узла к первому
    y[i] = d[i] - c[i] * y[i+1]  # Вычисление y_i через y_{i+1}


y[0] = 0.0   # Левое граничное условие y(0) = 0
y[-1] = 0.0  # Правое граничное условие y(1) = 0 

# Аналитическое решение для проверки
import math
sqrt_5_5 = math.sqrt(5.5)  # Вычисление константы
C2 = 1 / (5.5 * math.sin(sqrt_5_5))  # Коэффициент аналитического решения
analytical_solution = lambda x: C2 * np.sin(sqrt_5_5 * x) - (1/5.5)*x


for i in range(n+1):  # Для всех узлов сетки
    print(f"x = {x_nodes[i]:.3f}, y_num = {y[i]:.6f}, y_analytical = {analytical_solution(x_nodes[i]):.6f}")
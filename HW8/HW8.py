import numpy as np
from scipy.integrate import quad

# Определение базисных функций a_i(x) и b_i(y)
def a1(x): 
    return 1

def a2(x): 
    return -x**2 / 2

def a3(x): 
    return x**4 / 24

def b1(y): 
    return 1

def b2(y): 
    return y**4

def b3(y): 
    return y**8

# Правая часть уравнения f(x)
def f_x(x):
    # Для x==0 функция определена отдельно, чтобы избежать деления на ноль.
    if x == 0:
        return (0 - 1) * 0.45
    else:
        return (2*x - np.sin(x)/x) * 0.35

#--- Вычисление матрицы A для вырожденного ядра ---
# Используем размерность, равную числу базисных функций: 3.
n_funcs = 3
A = np.zeros((n_funcs, n_funcs))
for i in range(n_funcs):
    for j in range(n_funcs):
        a_func = [a1, a2, a3][i]
        b_func = [b1, b2, b3][j]
        A[i, j], _ = quad(lambda x: a_func(x) * b_func(x), 0, 1)

print("Матрица A:")
print(np.round(A, 4))
print("\n")

#--- Построение вырожденного ядра K_3(x, y) ---
def K_approx(x, y):
    return a1(x)*b1(y) + a2(x)*b2(y) + a3(x)*b3(y)

# Определение сетки для вывода значений ядра с шагом 0.1
x_grid = np.arange(0, 1.01, 0.1)
y_grid = np.arange(0, 1.01, 0.1)

print("Значения вырожденного ядра K_3(x, y) на сетке:")
for x in x_grid:
    for y in y_grid:
        print(f"K({x:.1f}, {y:.1f}) = {K_approx(x, y):.4f}")
    print()

#--- Вычисление вектора f_i ---
f = np.array([
    quad(lambda y: b1(y) * f_x(y), 0, 1)[0],
    quad(lambda y: b2(y) * f_x(y), 0, 1)[0],
    quad(lambda y: b3(y) * f_x(y), 0, 1)[0]
])

print(f"f = {f}")

#--- Решение системы (I - A)c = f ---
I = np.eye(n_funcs)
c = np.linalg.solve(I - A, f)

print(f"c = {c}")

#--- Приближенное решение ---
def u_approx(x):
    return f_x(x) - (c[0]*a1(x) + c[1]*a2(x) + c[2]*a3(x))

# Сетка для вывода приближенных значений u(x): шаг 0.1 (11 точек)
x_values = np.arange(0, 1.01, 0.1)
u_values = [u_approx(x) for x in x_values]

print("Приближенные значения u(x):")
for x, u in zip(x_values, u_values):
    print(f"x = {x:.1f}: {u:.3f}")

# Оценка погрешности (закомментированная часть)
# gamma = 0.4
# error_estimate = gamma**3
# print(f"\nОценка погрешности (γ^n, γ=0.4, n=3): {error_estimate:.3f}")

import numpy as np
from scipy.integrate import quad

# Определение функций a_i(x) и b_i(y)
def a1(x): return 1
def a2(x): return -x**2 / 2
def a3(x): return x**4 / 24
def b1(y): return 1
def b2(y): return y**4
def b3(y): return y**8

# Правая часть уравнения f(x)
def f_x(x):
    if x == 0:
        return (0 - 1) * 0.45
    else:
        return (2*x - np.sin(x)/x) * 0.35


n = 3
# Вычисление матрицы A
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a_func = [a1, a2, a3][i]
        b_func = [b1, b2, b3][j]
        A[i,j], _ = quad(lambda x: a_func(x) * b_func(x), 0, 1)

# Вывод матрицы A
print("Матрица A:")
print(np.round(A, 4))  # Округление до 4 знаков после запятой
print("\n")

# Построение вырожденного ядра K_3(x, y)
def K_approx(x, y):
    return a1(x)*b1(y) + a2(x)*b2(y) + a3(x)*b3(y)

# Вывод значений ядра на сетке [0, 1]x[0, 1]
x_grid = np.linspace(0, 1, n)
y_grid = np.linspace(0, 1, n)
print("Значения вырожденного ядра K_3(x, y) на сетке:")
for x in x_grid:
    for y in y_grid:
        print(f"K({x:.1f}, {y:.1f}) = {K_approx(x, y):.4f}")
    print()


# Вычисление вектора f_i
f = np.array([
    quad(lambda y: b1(y) * f_x(y), 0, 1)[0],
    quad(lambda y: b2(y) * f_x(y), 0, 1)[0],
    quad(lambda y: b3(y) * f_x(y), 0, 1)[0]
])

print(f"f = {f}")

# Решение системы (I - A)c = f
I = np.eye(3)
c = np.linalg.solve(I - A, f)

print(f"c = {c}")

# Приближенное решение
def u_approx(x):
    return f_x(x) - (c[0]*a1(x) + c[1]*a2(x) + c[2]*a3(x))

# Значения на сетке
x_values = np.linspace(0, 1, n)
u_values = [u_approx(x) for x in x_values]

print("Приближенные значения u(x):")
for x, u in zip(x_values, u_values):
    print(f"x = {x:.1f}: {u:.3f}")

# # Оценка погрешности
# gamma = 0.4
# error_estimate = gamma**3
# print(f"\nОценка погрешности (γ^n, γ=0.4, n=3): {error_estimate:.3f}")
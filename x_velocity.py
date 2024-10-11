import sympy as sp

import numpy as np
import matplotlib.pyplot as plt


y, x = sp.symbols('y x')

# omega = 1
# b = 0.5

omega, b = sp.symbols('omega b', real=True)
ksi_x = sp.symbols('ksi_x', real=True)
i = sp.I


def solve_x_velocity():
    # Определяем ksi(x) и ее производную по x
    ksi = sp.Function('ksi')(x)

    # ksi = 0.1 * sp.cos(x)
    ksi_prime = ksi.diff(x)  # Первая производная по x

    U = sp.Function('U')(y)

    # Уравнение с первой производной ksi'(x)
    eq = sp.Eq(U.diff(y, 2) - i * U * omega, i * b * omega**2 * ksi_prime)

    general_solution = sp.dsolve(eq, U)
    U_general = general_solution.rhs

    C1, C2 = sp.symbols('C1 C2')

    dU_dy = U_general.diff(y)

    boundary_condition_1 = sp.Eq(U_general.subs(y, 0), 0)  # U(0) = 0
    boundary_condition_2 = sp.Eq(
        dU_dy.subs(y, ksi_prime), 0
    )  # dU/dy |_{y=ksi'(x)} = 0

    constants_solution = sp.solve(
        (boundary_condition_1, boundary_condition_2), (C1, C2)
    )
    U_particular = U_general.subs(constants_solution)

    # Можно также взять производную по x
    # dU_dx = U_particular.diff(x)

    return U_particular


def plot_U(U_particular):
    U_numeric = sp.lambdify((x, y), U_particular, 'numpy')

    # Создание сетки значений x и y
    x_values = np.linspace(0, 6, 100)
    y_values = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x_values, y_values)

    # Вычисление значений U на сетке
    U_values = U_numeric(X, Y)

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.real(U_values), levels=100, cmap='viridis')
    plt.colorbar(label='Re(U)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Поле функции U(x, y)')
    plt.show()


if __name__ == '__main__':
    print(solve_x_velocity())
    # print(plot_U(solve_x_velocity()))

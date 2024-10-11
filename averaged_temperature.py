import sympy as sp

import numpy as np
import matplotlib.pyplot as plt


y, x, ksi, Bi = sp.symbols('y x ksi Bi')


def solve_T_averaged_Dirichlet_boundary():
    T = sp.Function('T')(y)

    eq = sp.Eq(T.diff(y, 2), 0)

    general_solution = sp.dsolve(eq, T)

    # делаем константы символьными переменными
    C1 = sp.symbols('C1')
    C2 = sp.symbols('C2')

    # Извлекаем выражение для T(y) из общего решения
    T_general = general_solution.rhs

    # Граничные условия
    boundary_1 = sp.Eq(T_general.subs(y, 0), -1)  # T(0) = -1
    boundary_2 = sp.Eq(
        T_general.subs(y, ksi), Bi * T_general.subs(y, ksi)
    )  # T(ksi) = -Bi * T(ksi)

    # Решаем систему уравнений для C1 и C2
    constants_solution = sp.solve([boundary_1, boundary_2], [C1, C2])

    T_particular = T_general.subs(constants_solution)

    return T_particular


def solve_T_averaged_Neuman_boundary():
    T = sp.Function('T')(y)

    # Уравнение
    eq = sp.Eq(T.diff(y, 2), 0)

    # Общее решение
    general_solution = sp.dsolve(eq, T)
    T_general = general_solution.rhs

    # Граничные условия для производной
    dT_dy = T_general.diff(y)  # Производная T(y) по y

    # Новые граничные условия
    boundary_1 = sp.Eq(dT_dy.subs(y, 0), -1)  # dT/dy(0) = -1
    boundary_2 = sp.Eq(
        dT_dy.subs(y, ksi), -Bi * T_general.subs(y, ksi)
    )  # dT/dy(ksi) = -Bi * T(ksi)

    # Решение системы уравнений для C1 и C2
    constants_solution = sp.solve(
        [boundary_1, boundary_2], sp.symbols('C1 C2')
    )

    # Подставляем найденные константы в общее решение
    T_particular = T_general.subs(constants_solution)
    return T_particular


def plot_T(T_particular):
    ksi_expr = 0.1 * sp.cos(x)
    T_particular_x_y = T_particular.subs(ksi, ksi_expr)

    # Численная функция
    T_numeric = sp.lambdify((x, y, Bi), T_particular_x_y, 'numpy')

    x_vals = np.linspace(0, 10, 100)
    y_vals = np.linspace(0, 1, 100)

    Bi_val = 1

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = T_numeric(X, Y, Bi_val)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(cp)

    plt.title('Temperature')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    plot_T(solve_T_averaged_Neuman_boundary())

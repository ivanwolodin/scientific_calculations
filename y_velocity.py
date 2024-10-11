import sympy as sp

import numpy as np
import matplotlib.pyplot as plt

from x_velocity import solve_x_velocity


y, x = sp.symbols('y x')

# omega = 1
# b = 0.5

omega, b = sp.symbols('omega b', real=True)
# ksi_x = sp.symbols('ksi_x', real=True)
i = sp.I


def solve_y_velocity():
    U_particular = solve_x_velocity()
    dU_dx = U_particular.diff(x)

    # V_general = sp.integrate(dU_dx, y) + sp.symbols('C')  # Добавляем константу интегрирования C
    #
    # C_solution = sp.solve(V_general.subs(y, 0), sp.symbols('C'))
    #
    # V_general = V_general.subs(sp.symbols('C'), C_solution[0])
    V_general = sp.integrate(dU_dx, y) + sp.symbols('C')

    return V_general


def plot_V(V_particular):
    V_numeric = sp.lambdify((x, y), V_particular, 'numpy')

    x_values = np.linspace(0, 6, 100)
    y_values = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x_values, y_values)

    V_values = V_numeric(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.real(V_values), levels=100, cmap='viridis')
    plt.colorbar(label='Re(V)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Поле функции V(x, y)')
    plt.show()


if __name__ == '__main__':
    print(solve_y_velocity())
    # plot_V(solve_y_velocity())

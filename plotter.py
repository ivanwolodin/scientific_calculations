import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sqrt, exp, cos, I, lambdify
from scipy.integrate import cumulative_trapezoid

output_dir = './res'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

y, x, t = symbols('y x t', real=True)
b = 1.0
omega = 10.0

lamda = (sqrt(omega) / sqrt(2)) * (1 + I)
ksi = 0.9 + 0.1 * cos(x)

ksi_prime = diff(ksi, x)
ksi_double_prime = diff(ksi_prime, x)

A = (b * omega * ksi_prime) / (1 + exp(2 * lamda * ksi)) * (
        exp(lamda * y) + exp(-lamda * y) * exp(2 * lamda * ksi)
) - (b * omega * ksi_prime)

B = (
        b
        * omega
        * (
                ksi_prime ** 2
                * (2 * exp(lamda * ksi) * (exp(lamda * y) + exp(-lamda * y) - 1))
                / (1 + exp(2 * lamda * ksi)) ** 2
                - ksi_double_prime
                * (exp(lamda * y) - exp(2 * lamda * ksi * (exp(-lamda * y) + 1)))
                / (lamda * (1 + exp(2 * lamda * ksi)))
                + y * ksi_double_prime
        )
)

U_real = (A * exp(I * t)).expand(complex=True).as_real_imag()[0]
V_real = (B * exp(I * t)).expand(complex=True).as_real_imag()[0]

# Преобразование в численные функции
U_func = lambdify((x, y, t), U_real, 'numpy')
V_func = lambdify((x, y, t), V_real, 'numpy')

x_vals = np.linspace(0, 20, 2000)
y_vals = np.linspace(0, 1.1, 2000)
X, Y = np.meshgrid(x_vals, y_vals)

# Линия y = 1.0 + 0.1 * cos(x)
y_limit = 0.9 + 0.1 * np.cos(x_vals)

# Моменты времени
times = [0, 0.252 * np.pi, 0.51 * np.pi, 0.752 * np.pi, np.pi]

for i, t_val in enumerate(times):
    fig, ax = plt.subplots(figsize=(6, 5))

    U_plot = U_func(X, Y, t_val)

    psi_U = cumulative_trapezoid(U_plot, y_vals, axis=0, initial=0)

    # Обнуляем значения psi выше линии y_limit
    for j in range(len(x_vals)):
        psi_U[:, j][Y[:, j] > y_limit[j]] = np.nan

    # Построение контура
    cs = ax.contour(X, Y, psi_U, levels=30, cmap='plasma')

    # Подпись каждой 5-й линии
    ax.clabel(cs, cs.levels[::1], inline=True, fontsize=8)

    # ax.set_title(f't = {t_val:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(0, 6.1)
    ax.set_ylim(0, 1.01)

    # Добавить линию
    ax.plot(x_vals, y_limit, 'k--', label='y = 0.1 cos(x)')

    # Сохранение графика для каждого момента времени
    filename = os.path.join(output_dir, f'omega={omega}_b={b}_t={t_val:.2f}.png')
    plt.savefig(filename)
    plt.close(fig)

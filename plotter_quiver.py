import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, cos, exp, sqrt, I, lambdify

# Символьные переменные
y, x, t = symbols('y x t')
b = 10.0
omega = 1.0

lamda = (sqrt(omega) / sqrt(2)) * (1 + I)

ksi = 0.1 * cos(x)
ksi_prime = diff(ksi, x)
ksi_double_prime = diff(ksi_prime, x)

A = (b * omega * ksi_prime) / (1 + exp(2 * lamda * ksi)) * (
    exp(lamda * y) + exp(-lamda * y) * exp(2 * lamda * ksi)
) - (b * omega * ksi_prime)

B = (
    b
    * omega
    * (
        ksi_prime**2
        * (2 * exp(lamda * ksi) * (exp(lamda * y) + exp(-lamda * y) - 1))
        / (1 + exp(2 * lamda * ksi)) ** 2
        - ksi_double_prime
        * (exp(lamda * y) - exp(2 * lamda * ksi * (exp(-lamda * y) + 1)))
        / (lamda * (1 + exp(2 * lamda * ksi)))
        + y * ksi_double_prime
    )
)

U_expr = A * exp(I * t)
V_expr = B * exp(I * t)

U_func = lambdify((x, y, t), U_expr, 'numpy')
V_func = lambdify((x, y, t), V_expr, 'numpy')

x_vals = np.linspace(0, 7, 60)
y_vals = np.linspace(0, 1, 60)
X, Y = np.meshgrid(x_vals, y_vals)


def plot_vector_field(t_val):
    U_vals = np.real(U_func(X, Y, t_val))
    V_vals = np.real(V_func(X, Y, t_val))

    # plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U_vals, V_vals, color='r')
    plt.title(f'Векторное поле скорости при t = {t_val}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


time_values = [0, 0.25 * np.pi, 0.62 * np.pi, 0.75 * np.pi]
for t_val in time_values:
    plot_vector_field(t_val)

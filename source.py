from sympy import *
from scipy.optimize import linprog
import numpy as np
import copy
import re


def get_grad_in_point_as_expr(expr_param, point):
    grad = sympify(0)
    variables = sorted(list(expr_param.atoms(Symbol)), key=lambda sym: sym.name)
    substitutions = []
    for i in range(0, len(variables)):
        substitutions.append((variables[i], point[i]))
    for i in range(0, len(variables)):
        variable = variables[i]
        derivative = diff(expr_param, variable)
        grad = grad + derivative.subs(substitutions) * variable
    return grad


def get_a_for_frank_wolfe(expr_param, h, xk):
    a = symbols('a')
    a_vec = np.array([])
    for i in range(0, len(h)):
        a_vec = np.append(a_vec, xk[i] + h[i] * a)
    expr_with_a = copy.deepcopy(expr_param)
    expr_variables = sorted(list(expr_with_a.atoms(Symbol)), key=lambda sym: sym.name)
    for i in range(0, len(expr_variables)):
        expr_with_a = expr_with_a.subs(expr_variables[i], a_vec[i])
    derivative = diff(expr_with_a, a)
    a_solve = solve(derivative, a)
    if len(a_solve) == 0:
        return 0
    a = float(solve(derivative, a)[0])
    if a > 1:
        return 1
    else:
        return a


def is_lim_reached(current_x, next_x, min_diff):
    count = 0
    for i in range(0, len(current_x)):
        if abs(current_x[i] - next_x[i]) < min_diff:
            count += 1
    if count == len(current_x):
        return True
    else:
        return False


def get_min_by_frank_wolfe(expr_param, A_param, b_param, bounds, x0, min_diff):
    current_x = x0
    next_x = np.array([])
    for i in range(0, len(current_x)):
        next_x = np.append(next_x, current_x[i] + min_diff + 1)

    while not is_lim_reached(current_x, next_x, min_diff):
        grad = get_grad_in_point_as_expr(expr_param, current_x)
        grad_coefficients = sorted(grad.as_coefficients_dict().items(), key=lambda item: item[0].name)
        c = []
        for i in grad_coefficients:
            c.append(i[1])
        linprog_res = linprog(c, A_ub=A_param, b_ub=b_param, bounds=bounds, method='highs')
        h = np.array(linprog_res.x) - current_x
        a = get_a_for_frank_wolfe(expr_param, h, current_x)
        next_x = current_x + a * h
        temp = next_x
        next_x = current_x
        current_x = temp
    return current_x


def get_max_by_frank_wolfe(expr_param, A_param, b_param, bounds, x0, min_diff):
    current_x = x0
    next_x = np.array([])
    for i in range(0, len(current_x)):
        next_x = np.append(next_x, current_x[i] + min_diff + 1)

    while not is_lim_reached(current_x, next_x, min_diff):
        grad = get_grad_in_point_as_expr(expr_param, current_x)
        grad_coefficients = sorted(grad.as_coefficients_dict().items(), key=lambda item: item[0].name)
        c = []
        for i in grad_coefficients:
            c.append(i[1])
        linprog_res = linprog(c, A_ub=A_param, b_ub=b_param, bounds=bounds, method='highs')
        h = np.array(linprog_res.x) - current_x
        a = get_a_for_frank_wolfe(expr_param, h, current_x)
        next_x = current_x + a * h
        temp = next_x
        next_x = current_x
        current_x = temp
    return current_x


def get_max_by_utopia_point_method(criteria, A_param, b_param, bounds=((0, None), (0, None)), x0=np.array([0, 0]), min_diff=0.01):
    utopia_values = np.array([])
    for i in criteria:
        c = []
        coefficients = sorted(i.as_coefficients_dict().items(), key=lambda item: item[0].name)
        for j in coefficients:
            c.append(-1 * j[1])
        linprog_res = linprog(c, A_ub=A_param, b_ub=b_param, bounds=bounds, method='highs')
        utopia_values = np.append(utopia_values, -1 * linprog_res.fun)
    metric_func_expr = sympify(0)
    for i in range(0, len(criteria)):
        metric_func_expr += (criteria[i] - utopia_values[i])**2
    return get_min_by_frank_wolfe(metric_func_expr, A, b, bounds, x0, min_diff)


# x1, x2 = symbols('x1 x2')
# criteria_array = np.array([
#     x1 + x2,
#     -3*x1 + x2,
#     x1 - 3*x2
# ])
# A = [[-1, -1],
#      [1, -2],
#      [-3, 2],
#      [1, 0],
#      [0, 1]]
# b = [-20, 10, 20, 40, 30]
# initial_point = np.array([40, 30])
#
# print(get_max_by_utopia_point_method(criteria_array, A, b, x0=initial_point))

x1, x2 = symbols('x1 x2')
criteria_array = np.array([
    -x1 + 2*x2,
    2*x1 + x2,
    x1 - 3*x2
])
A = [[1, 1],
     [-1, 0],
     [1, 0],
     [0, -1],
     [0, 1]]
b = [6, -1, 3, -1, 4]
initial_point = np.array([0, 0])

print(get_max_by_utopia_point_method(criteria_array, A, b, x0=initial_point))

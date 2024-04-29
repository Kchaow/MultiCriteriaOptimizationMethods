from sympy import *
from scipy.optimize import linprog
import numpy as np
import copy
import re


def get_grad_in_point_as_expr(expr, point):
    grad = sympify(0)
    variables = list(expr.atoms(Symbol))
    for i in range(0, len(variables)):
        variable = variables[i]
        derivative = diff(expr, variable)
        grad = grad + derivative.subs(variable, point[i]) * variable
    return grad


def get_a_for_frank_wolfe(expr, h, xk):
    a = symbols('a')
    a_vec = np.array([])
    for i in range(0, len(h)):
        a_vec = np.append(a_vec, xk[i] + h[i] * a)
    expr_with_a = copy.deepcopy(expr)
    expr_variables = list(expr_with_a.atoms(Symbol))
    for i in range(0, len(expr_variables)):
        criteria_ind = int(re.findall("\\d+$", expr_variables[i].name)[0])
        expr_with_a = expr_with_a.subs(expr_variables[i], a_vec[criteria_ind - 1])
    derivative = diff(expr_with_a, a)
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


def get_max_by_frank_wolfe(expr, A, b, bounds, x0, min_diff):
    current_x = x0
    next_x = np.array([])
    for i in range(0, len(current_x)):
        next_x = np.append(next_x, current_x[i] + min_diff + 1)

    while not is_lim_reached(current_x, next_x, min_diff):
        grad = get_grad_in_point_as_expr(expr, current_x)
        grad_coefficients = grad.as_coefficients_dict()
        c = []
        for i in grad_coefficients:
            c.append(-1 * grad_coefficients[i])
        linprog_res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        h = np.array(linprog_res.x) - current_x
        a = get_a_for_frank_wolfe(expr, h, current_x)
        next_x = current_x + a * h
        temp = next_x
        next_x = current_x
        current_x = temp
    return current_x


x1, x2 = symbols('x1 x2')
expr = 4*x1 + 8*x2 - x1**2 - x2**2
A = [[1, 1], [1, -1]]
b = [3, 2]
print(get_max_by_frank_wolfe(expr, A, b, [(0, None), (0, None)], np.array([0, 0]), 0.01))


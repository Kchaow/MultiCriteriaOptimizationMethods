#
#   Метод главного критерия
#
from scipy.optimize import linprog


def get_max_by_threshold_optimization(c_matrix, A, b, other_c_b, main_c_index, boundaries):
    other_c_b_ind = 0
    c = []
    for i in c_matrix:
        if i != c_matrix[main_c_index]:
            i = [-1 * j for j in i]
            A.append(i)
            b.append(-1 * other_c_b[other_c_b_ind])
            other_c_b_ind += 1
        else:
            c = [-1 * j for j in i]
    return linprog(c=c, A_ub=A, b_ub=b, bounds=boundaries)


n = 10
c_p = [[1, 1],
       [-3, 1],
       [1, -3]]
A_p = [[-1, -1],
       [1, -2],
       [-3, 2],
       [1, 0],
       [0, 1]]
b_p = [-2*n, n, 2*n, 4*n, 3*n]
bound = [(0, None), (0, None)]
other_c_b_p = [-2*n, -6*n]
main_c = 0
res = get_max_by_threshold_optimization(c_p, A_p, b_p, other_c_b_p, main_c, bound)

print('Оптимальная точка: ', res.x)
print('Наибольшее значение: ', -1 * res.fun)

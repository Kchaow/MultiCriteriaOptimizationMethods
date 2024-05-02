#
#   Метод последовательных уступок
#
from scipy.optimize import linprog


def get_max_by_successive_concession_method(c, priorities, A, b, boundaries, delta):
    delta_ind = 0
    for i in range(0, len(priorities)-1):
        c[priorities[i]] = [-1 * j for j in c[priorities[i]]]
        res = linprog(c=c[priorities[i]], A_ub=A, b_ub=b, bounds=boundaries, method='highs-ds')
        if res.fun is None:
            b.append(0)
        else:
            b.append(res.fun+delta[delta_ind])
        A.append(c[priorities[i]])
        delta_ind += 1
    return linprog(c=[-1 * j for j in c[priorities[-1]]], A_ub=A, b_ub=b, bounds=boundaries)


n = 10
c_p = [[1, 1],
       [-3, 1],
       [1, -3]]
prior_p = [0, 1, 2]
b_p = [4*n, -1*n, 2*n, 1.5*n]
A_p = [[1, 2],
       [-2, -1],
       [1, 0],
       [0, 1]]
delta_p = [2, 1]
boundaries_p = [(0, None), (0, None)]
r = get_max_by_successive_concession_method(c=c_p, priorities=prior_p, boundaries=boundaries_p, A=A_p, b=b_p,
                                            delta=delta_p)

print('Оптимальная точка: ', r.x)

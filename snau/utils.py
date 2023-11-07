import numpy as np
from typing import Callable

#########################################################
## Exact linear system equations solvers
#########################################################

def gauss_to_upper(a: np.ndarray, f: np.ndarray):
    # uses gauss to get upper triangle matrix
    a = a.copy()
    f = f.copy()
    if len(a.shape) != 2 or len(a) != len(f) or a.shape[0] != a.shape[1]:
        raise ValueError(f"Wrong dimensions! {f.shape=} and {a.shape=} were provided")
    for cur_col in range(len((a))):
        max_row = a[cur_col:, cur_col].argmax() + cur_col
        a[[cur_col, max_row]] = a[[max_row, cur_col]] # swap
        f[[cur_col, max_row]] = f[[max_row, cur_col]]
        for i in range(cur_col+1, len(a)):
            coeff = a[i, cur_col] / a[cur_col, cur_col]
            a[i, :] -= a[cur_col, :] * coeff
            f[i] -= f[cur_col] * coeff
    return a, f

def solve_upper(a: np.ndarray, f: np.ndarray):
    # solve lse with upper triangle matrix
    if len(a.shape) != 2 or len(a) != len(f) or a.shape[0] != a.shape[1]:
        raise ValueError(f"Wrong dimensions! {f.shape=} and {a.shape=} were provided")
    x = np.zeros(len(a))
    for cur_col in range(len(a)-1, -1, -1):
        x[cur_col] = (f[cur_col] - np.dot(a[cur_col], x)) / a[cur_col, cur_col]
    return x

def solve_lower(a: np.ndarray, f: np.ndarray):
    # solve lse with lower triangle matrix
    if len(a.shape) != 2 or len(a) != len(f) or a.shape[0] != a.shape[1]:
        raise ValueError(f"Wrong dimensions! {f.shape=} and {a.shape=} were provided")
    x = np.zeros(len(a))
    for cur_col in range(len(a)):
        x[cur_col] = (f[cur_col] - np.dot(a[cur_col], x)) / a[cur_col, cur_col]
    return x

def solve_gauss(a: np.ndarray, f: np.ndarray):
    # uses gauss to get solution of equation
    a_up, f_up = gauss_to_upper(a, f)
    x = solve_upper(a_up, f_up)
    return x

def lu_decomposition(a: np.ndarray):
    # Decomposes matrix into L and U
    u = np.zeros(a.shape)
    l = np.zeros(a.shape)
    np.fill_diagonal(l, 1)
    for i in range(len(a)):
        for j in range(len(a)):
            if i <= j:
                u[i][j] = a[i][j] - np.dot(l[i, :i+1], u[:i+1, j])
            if i > j:
                l[i][j] = (a[i][j] - np.dot(l[i, :j+1], u[:j+1, j])) / u[j][j]
    return l, u

def solve_lu(a: np.ndarray, f: np.ndarray, l=None, u=None):
    # solves equation using lu_decomposition
    if l is None or u is None:
        l, u = lu_decomposition(a)
    y = solve_lower(l, f)
    return solve_upper(u, y)

#########################################################
## Iterative linear system equations solvers
#########################################################

def run_iter(a: np.ndarray, f: np.ndarray, iteration: Callable, x=None, iterations=10):
    if x is None:
        x = np.zeros(len(a))
    xs = []
    for i in range(iterations):
        x = iteration(a, f, x)
        xs.append(x)
    return xs
def seidel_iter(a: np.ndarray, f: np.ndarray, x: np.ndarray):
    x_new = np.copy(x)
    for i in range(len(a)):
        s1 = sum(a[i][j] * x_new[j] for j in range(i))
        s2 = sum(a[i][j] * x[j] for j in range(i + 1, len(a)))
        x_new[i] = (f[i] - s1 - s2) / a[i][i]
    return x_new
def jacobi_iter(a: np.ndarray, f: np.ndarray, x: np.ndarray):
    return np.diag(a)**(-1) * (f - np.dot(np.tril(a, -1) + np.triu(a, 1), x))
def relaxation_iter(a: np.ndarray, f: np.ndarray, x: np.ndarray, w=1):
    x_new = np.zeros(len(x))
    for k in range(len(x)):
        x_new[k] = x[k] + w/a[k][k] * (f[k] - np.dot(a[k], x))
    return x_new

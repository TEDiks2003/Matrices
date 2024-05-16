import numpy as np
import scipy.linalg as la
from Matrix import Matrix
from NumDiff import *


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Multiplies two matrices and returns new Matrix Object"""
    sum_int = m1.get_col_num
    assert sum_int == m2.get_row_num, "Matrices are incompatible for multiplication"
    rows = m1.get_row_num
    cols = m2.get_col_num

    arr_m1 = m1.get_arr
    arr_m2 = m2.get_arr

    output_arr = []

    for row_int in range(rows):
        row = []
        arr1 = arr_m1[row_int, :]
        for col_int in range(cols):
            arr2 = arr_m2[:, col_int]
            entry = 0
            for i in range(sum_int):
                entry += arr1[i] * arr2[i]
            row.append(entry)
        output_arr.append(row)

    return Matrix(np.array(output_arr))


def lu_decomposition(m: Matrix) -> (Matrix, Matrix, list[(int, int)]):
    mat = Matrix(m.get_arr.copy())
    lower_arr, upper_arr, swap_index = mat.lu_decomposition_self()
    return Matrix(lower_arr), Matrix(upper_arr), swap_index


def lu_decomposition_no_pp(m: Matrix) -> (Matrix, Matrix):
    mat = Matrix(m.get_arr.copy())
    lower_arr, upper_arr = mat.lu_decomposition_self_no_pp()
    return Matrix(lower_arr), Matrix(upper_arr)


def cholesky(mat: Matrix) -> Matrix:
    """Cholesky Decomposition"""
    arr = la.cholesky(mat.get_arr.copy(), lower=False)
    return Matrix(arr)


def qr_decomposition(mat: Matrix) -> (Matrix, Matrix):
    """QR decomposition"""
    q, r = la.qr(mat.get_arr.copy())
    return Matrix(q), Matrix(r)


def jacobian_derivatives(f_arr, x):
    h = 0.0003
    n = len(x)
    ret = []
    for f in f_arr:
        row = []
        for i in range(n):
            diff = np.zeros(n)
            diff[i] = h
            row.append((f(x+diff)-f(x-diff))/(2*h))
        ret.append(row)
    return np.array(ret)


def newton_non_linear(A, b, epsilon=0.000001, guess=None, print_work=False):
    """
    Solves non linear systems of equations Ax=b using Newton's Method
    """
    n = len(A)
    assert n == len(b), "A has to be the same length as b"
    if guess is None:
        guess = np.ones(1, n)

    assert len(guess) == n, "initial guess has to be the same length as b"

    f_arr = list(
        map(
            lambda i:
            lambda x: sum([A[i][j](x) * x[j] for j in range(n)]) - b[i],
            range(n)
        )
    )

    def f(x):
        """returns vector of f(x)=Ax-b"""
        return np.array([g(x) for g in f_arr])

    f_x = f(guess)

    while la.norm(f_x) > epsilon:
        if print_work:
            print(f"solution: {guess} error: {la.norm(f_x)}")
        j_x = jacobian_derivatives(f_arr, guess)
        delta_x = la.solve(j_x, (-1 * f_x).T)
        guess += delta_x
        f_x = f(guess)

    return guess

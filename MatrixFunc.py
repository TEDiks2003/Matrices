import numpy as np
import scipy.linalg as la
from Matrix import Matrix


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
                entry += arr1[i]*arr2[i]
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

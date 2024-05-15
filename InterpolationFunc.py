import numpy as np
import scipy as sc
import typing as typ

np_farr = typ.TypeVar("np_farr", bound="np.array(typ.Any, np.float64)")
np_polynomial = typ.TypeVar("np_polynomial", bound="np.poly1d")


def vandermonde(x_arr: np_farr, y_arr: np_farr) -> np_farr:
    """Interpolate coefficients of polynomial with inputs x and outputs b,
    output is from the highest order coefficients,
    Prone to roundoff error, edit data accordingly"""
    n = len(x_arr)
    a_mat = np.vander(x_arr, n)
    ret = np.linalg.solve(a_mat, y_arr)
    return ret


def lagrange(x_arr: np_farr, y_arr: np_farr) -> np_polynomial:
    """Interpolates Polynomial using Lagrange Coefficients"""
    poly = sc.interpolate.lagrange(x_arr, y_arr)
    return poly


def lagrange_coefficients(x_arr: np_farr, y_arr: np_farr) -> list[np_polynomial]:
    ret = []
    n = len(x_arr)
    for k in range(n):
        l_k = np.poly1d([1])
        for j in range(n):
            if j != k:
                c_1 = 1/(x_arr[k]-x_arr[j])
                c_2 = -x_arr[j]*c_1
                p = np.poly1d([c_1, c_2])
                l_k = np.polymul(l_k, p)
        ret.append(l_k)
    return ret


def p_n_min_1(a_arr: np_farr, w_arr: list[np_polynomial], x: np.float64) -> np.float64:
    ret = np.float64(0)
    for i in range(len(a_arr)):
        ret += a_arr[i]*w_arr[i](x)
    return ret


def newton_basis(x_arr: np_farr, y_arr: np_farr) -> np_polynomial:
    n = len(x_arr)

    w_arr = [np.poly1d([1])]
    for i in range(0, n):
        mult = np.poly1d([1, -x_arr[i]])
        w = np.polymul(w_arr[i], mult)
        w_arr.append(w)

    div_dif = [y_arr]
    for i in range(1, n):
        f_i = []
        for k in range(n-i):
            top_dif = div_dif[i-1][k+1]-div_dif[i-1][k]
            bottom_dif = x_arr[k+i]-x_arr[k]
            f_i_x_k = top_dif/bottom_dif
            f_i.append(f_i_x_k)
        div_dif.append(f_i)

    a_arr = []
    for i in range(n):
        a_arr.append(div_dif[i][0])

    print(a_arr)

    ret = np.poly1d([0])
    for i in range(n):
        a_poly = np.poly1d([a_arr[i]])
        to_add = np.polymul(a_poly, w_arr[i])
        ret = np.polyadd(ret, to_add)
    return ret


def piecewise_linear_spline(x_arr: np_farr, y_arr: np_farr) -> list[np_polynomial]:
    ret = []
    n = len(x_arr)
    for i in range(n-1):
        p = np.poly1d([1, -x_arr[i]])
        c = (y_arr[i+1]-y_arr[i])/(x_arr[i+1]-x_arr[i])
        p = np.polymul(p, np.poly1d([c]))
        f_x_i = np.poly1d([y_arr[i]])
        ret.append(np.polyadd(p, f_x_i))

    return ret


def cubic_spline(x_arr: np_farr, y_arr: np_farr) -> list[np_polynomial]:
    pass
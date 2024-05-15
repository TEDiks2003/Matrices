import math

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


def cubic_spline_manual(x_arr: np_farr, y_arr: np_farr, s_d_0: float, s_d_n: float, clamped: bool = True) -> list[np_polynomial]:
    """s_d_0 = S'(x_0), s_d_n = S'(x_N), ONLY WORKS FOR 4 DATA POINTS"""
    n = len(x_arr)
    h_k = [x_arr[i+1]-x_arr[i] for i in range(n-1)]
    d_k = [(y_arr[i+1]-y_arr[i])/h_k[i] for i in range(n-1)]
    u_k = [6*(d_k[i]-d_k[i-1]) for i in range(1, n-1)]

    assert n == 4, "We have a problem"

    if clamped:
        c_1 = 2*(h_k[0]+h_k[1])-(h_k[0]/2)
        c_2 = 2*(h_k[n-3]+h_k[n-2])-(h_k[n-2]/2)
        e_1 = -3*(d_k[0] - s_d_0)
        e_2 = -3*(s_d_n-d_k[n-2])
    else:
        c_1 = 2*(h_k[0]+h_k[1])
        c_2 = 2*(h_k[n-3]+h_k[n-2])
        e_1 = 0
        e_2 = 0

    row_1 = [c_1, h_k[1]]
    row_2 = [h_k[2], c_2]
    mat = np.array([row_1, row_2], dtype=np.float64)
    b = np.asfarray([u_k[0]+e_1, u_k[1]+e_2])

    sol = np.linalg.solve(mat, b)

    if clamped:
        m_0 = (3/h_k[0])*(d_k[0]-s_d_0)-(sol[0]/2)
        m_3 = (3/h_k[n-2])*(d_k[n-2])-(sol[1]/2)
    else:
        m_0 = 0
        m_3 = 0

    m_k = [m_0, sol[0], sol[1], m_3]

    s_k_1 = y_arr
    s_k_2 = [d_k[i]-(h_k[i]*(2*m_k[i]+m_k[i+1]))/6 for i in range(n-1)]
    s_k_3 = [m_k[i]/2 for i in range(n-1)]
    s_k_4 = [(m_k[i+1]-m_k[i])/(6*h_k[i]) for i in range(n-1)]

    for i in range(n-1):
        print(f"s_{i}(x) = {s_k_4[i]}(x-{x_arr[i]})^3+{s_k_3[i]}(x-{x_arr[i]})^2+{s_k_2[i]}(x-{x_arr[i]})+{s_k_1[i]}")

    g_k_1 = [np.poly1d([1, -x]) for x in x_arr]
    g_k_2 = [np.polymul(g, g) for g in g_k_1]
    g_k_3 = [np.polymul(g, h) for g, h in zip(g_k_1, g_k_2)]
    ret = []
    for i in range(n-1):
        a_0 = np.poly1d([s_k_1[i]])
        a_1 = np.poly1d([s_k_2[i]])
        a_2 = np.poly1d([s_k_3[i]])
        a_3 = np.poly1d([s_k_4[i]])
        b = np.polymul(a_1, g_k_1[i])
        c = np.polymul(a_2, g_k_3[i])
        d = np.polymul(a_3, g_k_3[i])
        f_1 = np.polyadd(a_0, b)
        f_2 = np.polyadd(c, d)
        ret.append(np.polyadd(f_1, f_2))

    return ret


def cubic_spline_scipy(x_arr: np_farr, y_arr: np_farr, ends: ((float, float), (float, float))):
    return sc.interpolate.CubicSpline(x_arr, y_arr, bc_type=ends)


def nevilles_method(x_arr: np_farr, y_arr: np_farr, x: float) -> float:
    n = len(x_arr)
    p_k = [y_arr]
    for i in range(1, n):
        p_i = []
        for k in range(n-i):
            l_l = (x-x_arr[k+i])
            l_r = (p_k[i-1][k])
            r_l = (p_k[i-1][k+1])
            r_r = (x_arr[k]-x)
            top_dif = (l_l*l_r)+(r_l*r_r)
            bottom_dif = x_arr[k]-x_arr[k+i]
            p_i_x_k = top_dif/bottom_dif
            p_i.append(p_i_x_k)
        p_k.append(p_i)

    return p_k[-1][0]


def ridders_method(f, x_1: float, x_2: float, epsilon: float) -> float:
    y_1 = f(x_1)
    y_2 = f(x_2)
    is_pos_f = y_1 < 0

    x_3 = (x_1+x_2)/2
    y_3 = f(x_3)

    if abs(y_3) < epsilon:
        return x_3

    c = 1 if (y_1-y_2) > 0 else -1

    x_4 = x_3 + c*(x_3-x_1)*(y_3/math.sqrt(y_3**2-(y_1*y_2)))

    while abs(f(x_4)) > epsilon:
        x_k = [x_1, x_2, x_3, x_4]
        neg = [x for x in x_k if f(x) < 0]
        pos = [x for x in x_k if f(x) > 0]
        if is_pos_f:
            x_1 = max(neg)
            x_2 = min(pos)
        else:
            x_2 = min(neg)
            x_1 = max(pos)
        x_3 = (x_1+x_2)/2
        if abs(f(x_3)) < epsilon:
            return x_3
        c = 1 if (f(y_1) - f(y_2)) > 0 else -1
        x_4 = x_3 + c * (x_3 - x_1) * (f(x_3) / math.sqrt(f(x_3) ** 2 - (f(x_1) * f(x_2))))
        print(f"x4 = {x_4}")

    return x_4

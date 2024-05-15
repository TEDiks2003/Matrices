def ivp_euler(d_f, x: int, f_x: int, interval: (float, float), m: int):
    """Initial Value Problem, given f' on an interval and f(x)=f_x find constant in integral of f'"""
    a, b = interval
    h = (b-a)/m
    t_k = [a+k*h for k in range(m+1)]
    y_k = [f_x]
    for i in range(1, m+1):
        y_i = y_k[i-1]+h*d_f(t_k[i-1], y_k[i-1])
        y_k.append(y_i)

    return y_k


def ivp_runge_kutta(d_f, x: int, f_x: int, interval: (float, float), m: int):
    """Initial Value Problem, given f' on an interval and f(x)=f_x find constant in integral of f'"""
    a, b = interval
    h = (b - a) / m
    t_k = [a + k * h for k in range(m + 1)]
    y_k = [f_x]
    for i in range(1, m + 1):
        f_1 = d_f(t_k[i - 1], y_k[i - 1])
        f_2 = d_f(t_k[i - 1]+h/2, y_k[i - 1]+h/2*f_1)
        f_3 = d_f(t_k[i - 1] + h / 2, y_k[i - 1] + h / 2 * f_2)
        f_4 = d_f(t_k[i - 1] + h, y_k[i - 1] + h * f_3)
        y_i = y_k[i - 1] + (h*(f_1+2*f_2+2*f_3+f_4))/6
        y_k.append(y_i)

    return y_k


def ivp_runge_kutta_dif_n(d_f_n, f_x_list: list[(float, float)], interval: (float, float), m: int, n: int):
    """Runge Kutta for higher order derivatives"""
    a, b = interval
    h = (b - a) / m
    t_k = [a + k * h for k in range(m + 1)]
    pass

def bvp_finite_difference_method(u, a, b, f_a, f_b, x_arr):
    pass

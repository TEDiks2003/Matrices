def dif_central_formulae(f, x: float, derivative: int = 1, order_of_h: int = 2, h: float = 0.0003):
    assert order_of_h in [2, 4], "order of h must be 2 or 4"
    assert derivative in [1, 2, 3], "order of derivative must be in [1, 2, 3]"
    if derivative == 1:
        if order_of_h == 2:
            d_f = (f(x+h)-f(x-h))/(2*h)
        else:
            d_f = (-f(x+2*h)+(8*f(x+h))-(8*f(x-h))+f(x-2*h))/(12*h)
    elif derivative == 2:
        if order_of_h == 2:
            d_f = (f(x+h)-(2*f(x))+f(x-h))/(h**2)
        else:
            d_f = (-f(x+2*h)+(16*f(x+h))-(30*f(x))+(16*f(x-h))-f(x-2*h))/(12*(h**2))
    else:
        if order_of_h == 2:
            d_f = (f(x+2*h)-(2*f(x+h))+(2*f(x-h))-f(x-2*h))/(2*h)
        else:
            d_f = (-f(x+3*h)+(8*f(x+2*h))-(13*f(x+h))+(13*f(x-h))-(8*f(x-2*h))+f(x-3*h))/(8*(h**2))

    return d_f

import numpy as np
from Matrix import *
from MatrixFunc import *


def program():
    m1 = Matrix("-3,2,-1;6,-6,7;3,-4,4;")
    print(m1)
    lower, upper, swaps = lu_decomposition(m1)
    m1.swap_rows_from_arr(swaps)
    print(lower)
    print(upper)
    print(m1)
    print(matrix_multiply(lower, upper))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

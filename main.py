import numpy as np
from Matrix import *
from MatrixFunc import *


def program():
    m1 = Matrix("2,3,-2,3;4,2,-1,9;4,-8,2,-6;")
    print(m1)
    m1.solve_partial_pivoting()
    print(m1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

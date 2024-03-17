import numpy as np
from Matrix import *
from MatrixFunc import *

def program():
    m1 = Matrix("7,4,9;2,1,1;")
    m2 = Matrix("5,3;3,-3;-8,7;")
    m3 = matrix_multiply(m1, m2)
    print(m3.get_arr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

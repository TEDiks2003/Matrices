import numpy as np
from Matrix import *
from MatrixFunc import *
import scipy as sc


def program():
    m1 = Matrix("9,1,1,10;2,10,3,19;3,4,11,0;")
    print(m1)
    arr = m1.jacobi([0.0, 0.0, 0.0], 0.0001)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

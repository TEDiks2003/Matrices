import numpy as np
from Matrix import *
from MatrixFunc import *
import scipy as sc


def program():
    m1 = Matrix("9,1,1;2,10,3;3,4,11;")
    print(m1)
    m1.append_b(np.asfarray([10.0, 19.0, 0.0]))
    print(m1)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

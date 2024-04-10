import numpy as np
from Matrix import *
from MatrixFunc import *
import scipy as sc


def program():
    m1 = Matrix("1,1;2,1;3,1;")
    print(m1)
    q, r = qr_decomposition(m1)
    print(q)
    print(r)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

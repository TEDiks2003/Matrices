import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
from InterpolationFunc import *
import scipy as sc
from random import sample
from NumDiff import *


def program():

    f = lambda x: math.cos(x)
    y = 0.8

    print(f"h^2: {dif_central_formulae(f, y, 2, 2)}, h^v: {dif_central_formulae(f, y, 2, 4)}")
    print(f(0.8))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

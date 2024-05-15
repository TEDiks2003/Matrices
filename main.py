import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
from InterpolationFunc import *
import scipy as sc
from random import sample


def program():

    f = lambda x: -(1/((x-0.3)**2+0.01)-1/((x-0.8)**2+0.04))
    print(ridders_method(f, 0.0, 1.0, 0.00000000001))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

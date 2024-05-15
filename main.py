import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
from InterpolationFunc import *
import scipy as sc
from random import sample


def program():

    x = np.asfarray([0, 1, 2, 3])
    y = np.asfarray([0, 0.5, 2.0, 1.5])
    cubic_spline_manual(x, y, 0.2, -1, False)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

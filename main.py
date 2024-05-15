import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
from InterpolationFunc import *
import scipy as sc
from random import sample


def program():

    x = np.asfarray([4.65, 4.70, 4.75, 4.80, 4.85])
    y = np.asfarray([-4.26066, -1.68111, 1.17327, 4.31638, 7.76207])
    f_x = nevilles_method(y, x, 0)
    print(f_x)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

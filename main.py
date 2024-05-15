import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
from InterpolationFunc import *
import scipy as sc
from random import sample
from NumDiff import *
from ODE import *
import matplotlib.pyplot as plt


def program():

    p = np.poly1d([-1, 7, -7])
    q = np.poly1d([3, -3, 8])
    x = np.arange(20)
    y = p(x)
    y_2 = q(x)
    plt.plot(x, y, label="p")
    plt.plot(x, y_2, label="q")
    plt.legend()
    plt.grid()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

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
    A = [
        [lambda x: 1.4, lambda x: -1],
        [lambda x: x[0]-1.6, lambda x: -1],
    ]

    b = np.asfarray([0.6, 4.6])
    guess = np.asfarray([5, 5])

    print(newton_non_linear(A, b, guess=guess, print_work=True))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

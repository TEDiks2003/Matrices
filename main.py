import numpy as np
from Matrix import *
from MatrixFunc import *
from DLP import *
import scipy as sc


def program():
    mat = Matrix(
        '''
        1, 1, 1, 1, 0, 16;
        1, 2, 0, 0, 1, 10;
        1, 0, 0, 1, 1, 5;
        0, 1, 1, 0, 1, 12;
        2, 2, 1, 0, 0, 5;
        '''
    )
    print(mat)
    mat.optimise_for_solving()
    print(mat)
    print(mat.solve_mod_p(19))
    # dlp(13, 6, 229, [4, 3, 19])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

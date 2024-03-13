import numpy as np
from Matrix import *


def program():
    content = input("Matrix input: ")
    m1 = Matrix(content)
    print(m1.get_arr())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    program()

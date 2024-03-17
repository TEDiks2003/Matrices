import typing

import numpy as np


class Matrix:
    """Matrix Class with Matrix related Functions"""
    _row_num: int
    _col_num: int
    _arr: np.ndarray[typing.Any, np.float64] = None
    _determinant: float = None
    _is_square: bool

    def __init__(self, content: str | np.ndarray[typing.Any, np.float64]):
        """Content Formatted like: \"x_1_1, x_2_1 x_3_1; x_1_2, x_2_2 x_3_2;\" or a numpy array """
        if type(content) == str:
            row_counter = 0
            col_counter = 0
            col_num = 0

            for char in content:
                if char == ',':
                    col_counter += 1
                elif char == ';':
                    col_counter += 1
                    row_counter += 1
                    if col_num == 0:
                        col_num = col_counter
                        col_counter = 0
                    else:
                        assert col_counter == col_num, "Row's have different lengths!"
                        col_counter = 0

            self._col_num = col_num
            self._row_num = row_counter

            buffer = ""
            content_list = []
            current_list = []

            for char in content:
                if char == " ":
                    continue
                elif char == "," or char == ";":
                    current_list.append(buffer)
                    buffer = ""
                    if char == ";":
                        content_list.append(current_list)
                        current_list = []
                else:
                    buffer += char

            try:
                self._arr = np.asfarray(np.array(content_list))
            except Exception as e:
                print(e)
                exit()
        else:
            try:
                shape = content.shape
            except Exception as e:
                print("ERROR in constructing matrix from arr:", e)

            if len(shape) == 1:
                self._row_num = 1
                self._col_num = shape[0]
            else:
                self._row_num = shape[-2]
                self._col_num = shape[-1]
                self._arr = content

        self._is_square = self._row_num == self._col_num

    @property
    def get_arr(self) -> np.ndarray[typing.Any, np.float64]:
        """Return Numpy Array storing content"""
        return self._arr

    @property
    def get_col_num(self):
        """Return number of columns"""
        return self._col_num

    @property
    def get_row_num(self):
        """Return number of rows"""
        return self._row_num

    @property
    def get_determinant(self) -> float | None:
        """Return Matrix Determinant"""
        assert self._is_square, "Matrix is not square"

        if self._determinant is None:
            self._determinant = np.linalg.det(self._arr)
        return self._determinant

    def __str__(self):
        """Matrix array as string"""
        string = ""
        for row in self._arr:
            for entry in row:
                string += str(entry)+", "
            string = string[:-2]+";\n"
        return string

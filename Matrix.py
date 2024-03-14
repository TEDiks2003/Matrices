import typing

import numpy as np
from nptyping import NDArray


class Matrix:
    _row_num: int
    _col_num: int
    _arr: NDArray[typing.Any, np.float64]
    _determinant: float = None

    def __init__(self, content: str):

        # content format "x_1_1, x_2_1 x_3_1; x_1_2, x_2_2 x_3_2;"

        col_counter = 0
        row_counter = 0
        row_num = 0

        for char in content:
            if char == ',':
                row_counter += 1
            elif char == ';':
                row_counter += 1
                col_counter += 1
                if row_num == 0:
                    row_num = row_counter
                    row_counter = 0
                else:
                    assert row_counter == row_num, "Row's have different lengths!"
                    row_counter = 0

        self._row_num = row_num
        self._col_num = col_counter

        buffer = ""
        content_list = []
        current_list = []

        for char in content:
            if char == ' ':
                continue
            elif char == ',' or char == ';':
                current_list.append(buffer)
                buffer = ""
                if char == ';':
                    content_list.append(current_list)
                    current_list = []
            else:
                buffer += char

        try:
            self._arr = np.asfarray(np.array(content_list))
        except Exception as e:
            print(e)
            exit()

    def get_arr(self) -> NDArray[typing.Any, np.float64]:
        return self._arr

    def get_determinant(self) -> float:
        if self._determinant is None:
            self._determinant = np.linalg.det(self._arr)
        return self._determinant

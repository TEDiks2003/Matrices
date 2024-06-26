import typing

import numpy as np

from typing import TypeVar

Mat = TypeVar("Mat", bound="Matrix")


def gcd_extended(x: int, mod: int) -> (int, int, int):
    """Finds inverse of x"""
    if x == 0:
        return mod, 0, 1

    gcd, a1, b1 = gcd_extended(mod % x, x)
    a = b1 - (mod // x) * a1
    b = a1

    return gcd, a, b


class Matrix:
    """Matrix Class with Matrix related Functions"""
    _row_num: int
    _col_num: int
    _arr: np.ndarray[typing.Any, np.float64] = None
    _determinant: float = None
    _is_square: bool
    _is_linalg_system: bool

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
        self._is_linalg_system = self._row_num+1 == self._col_num

    @property
    def get_arr(self) -> np.ndarray[typing.Any, np.float64]:
        """Return Numpy Array storing content"""
        return self._arr

    @property
    def get_col_num(self) -> int:
        """Return number of columns"""
        return self._col_num

    @property
    def get_row_num(self) -> int:
        """Return number of rows"""
        return self._row_num

    @property
    def get_determinant(self) -> float | None:
        """Return Matrix Determinant"""
        assert self._is_square, "Matrix is not square"

        if self._determinant is None:
            self._determinant = np.linalg.det(self._arr)
        return self._determinant

    def __str__(self) -> str:
        """Matrix array as string"""
        string = ""
        for row in self._arr:
            for entry in row:
                string += str(entry) + ", "
            string = string[:-2] + ";\n"
        return string

    def _row_swap(self, row1: int, row2: int) -> None:
        """Swap two rows in Matrix"""
        assert row1 < self._row_num and row2 < self._row_num, "Row number out of index"
        self._arr[[row1, row2]] = self._arr[[row2, row1]]

    def _row_multiplication(self, row: int, c: np.float64) -> None:
        """Multiply row by constant"""
        assert row < self._row_num, "Row number out of index"
        vec = np.array([1.0 for i in range(self._row_num)])
        vec[row] = c
        self._multiply_rows_scalar(vec)

    def _multiply_rows_scalar(self, vec: np.ndarray[typing.Any, np.float64]) -> None:
        """Multiply rows by scalars from vector"""
        assert vec.shape == (self._row_num,), "wrong vector size for multiplying across rows"
        self._arr = (self._arr.T * vec).T

    def _row_add_row(self, row1: int, row2: int, c: np.float64) -> None:
        """Add rows: row1 = row1+c*row2"""
        assert row1 < self._row_num and row2 < self._row_num, "Row number out of index"

        self._arr[row1] = np.add(self._arr[row1], self._arr[row2]*c)

    def solve_partial_pivoting(self):
        """Solves System of equations using partial pivoting"""
        for k in range(self._row_num-1):
            index_to_swap = self._find_largest_coefficient(k)
            self._row_swap(k, index_to_swap)
            entry_value = self._arr[k][k]
            for i in range(k+1, self._row_num):
                c = self._arr[i][k]/entry_value
                self._row_add_row(i, k, -c)

        for k in range(self._row_num-1, -1, -1):
            val = self._arr[k][k]
            print(1/val)
            self._row_multiplication(k, (1/val))
            for i in range(k-1, -1, -1):
                val = self._arr[i][k]
                self._row_add_row(i, k, -val)

    def _find_largest_coefficient(self, k: int) -> int:
        """Finds largest coefficient for partial pivoting"""
        largest = 0
        index = -1
        for i in range(k, self._row_num):
            val = abs(self._arr[i][k])
            if val > largest:
                largest = val
                index = i

        assert index > 0, "Only found 0 coefficients"
        return index

    def lu_decomposition_self(self) -> (np.ndarray[typing.Any, np.float64], np.ndarray[typing.Any, np.float64], [(int, int)]):
        """Decomposes Matrix using LU decomposition"""
        swap_record = []
        lower = np.zeros((self._row_num, self._col_num), dtype=np.float64)
        for k in range(self._row_num-1):
            lower[k][k] = 1
            index_to_swap = self._find_largest_coefficient(k)
            swap_record.append((k, index_to_swap))
            self._row_swap(k, index_to_swap)
            entry_value = self._arr[k][k]
            for i in range(k+1, self._row_num):
                c = self._arr[i][k]/entry_value
                self._row_add_row(i, k, -c)
                lower[i][k] = c
        lower[self._row_num-1][self._row_num-1] = 1

        return lower, self._arr.copy(), swap_record

    def swap_rows_from_arr(self, arr: list[(int, int)]) -> None:
        """Performs swaps defined in swap record list"""
        for tup in arr:
            self._row_swap(tup[0], tup[1])

    def lu_decomposition_self_no_pp(self) -> (np.ndarray[typing.Any, np.float64], np.ndarray[typing.Any, np.float64]):
        """Decomposes Matrix using LU decomposition"""
        lower = np.zeros((self._row_num, self._col_num), dtype=np.float64)
        for k in range(self._row_num-1):
            lower[k][k] = 1
            entry_value = self._arr[k][k]
            for i in range(k+1, self._row_num):
                c = self._arr[i][k]/entry_value
                self._row_add_row(i, k, -c)
                lower[i][k] = c
        lower[self._row_num-1][self._row_num-1] = 1

        return lower, self._arr.copy()

    def jacobi(self, init: list[np.float64], epsilon: np.float64) -> list[np.float64]:
        """Jacobi Approximation given initial guess and absolute error epsilon"""
        assert self._is_linalg_system, "Not a system of linear equations!"
        assert len(init) == self._row_num, "Initial guess shape is incorrect"

        while True:
            print(init)
            biggest_err = 0
            new_init = []
            for i in range(len(init)):
                prev_app = init[i]
                new_app = self._sub_iteration_approx(self.get_arr[i], init, i)
                new_init.append(new_app)
                err = abs(new_app-prev_app)
                if err > biggest_err:
                    biggest_err = err
            init = new_init
            if biggest_err < epsilon:
                break

        return init

    def gauss_seidel(self, init: list[np.float64], epsilon: np.float64) -> list[np.float64]:
        """Gauss Seidel Approximation given initial guess and absolute error epsilon"""
        assert self._is_linalg_system, "Not a system of linear equations!"
        assert len(init) == self._row_num, "Initial guess shape is incorrect"

        while True:
            print(init)
            biggest_err = 0
            for i in range(len(init)):
                prev_app = init[i]
                new_app = self._sub_iteration_approx(self.get_arr[i], init, i)
                init[i] = new_app
                err = abs(new_app-prev_app)
                if err > biggest_err:
                    biggest_err = err
            if biggest_err < epsilon:
                break

        return init

    @staticmethod
    def _sub_iteration_approx(m_arr:  np.ndarray[typing.Any, np.float64], x:  list[np.float64], j: int) -> np.float64:
        """Iterates approximations m_arr is array row, x is current approximations, j is index of x_j being approximated"""
        assert len(m_arr) == len(x)+1, "Error!"
        ret = 0.0
        scalar = m_arr[j]
        m_arr = np.delete(m_arr, j)
        x = np.delete(x, j)
        for i in range(len(x)):
            c = -m_arr[i]
            x_i = x[i]
            ret += c*x_i
        ret += m_arr[-1]

        return ret/scalar

    def append_b(self, b: np.ndarray[typing.Any, np.float64]) -> None:
        """Append vector b to square matrix"""
        assert self._is_square, "Not a square matrix"
        assert b.shape == (self._row_num,), "Incorrect shape of b"

        self._arr = np.append(self._arr, np.array([b]).T, axis=1)
        self._col_num += 1
        self._is_square = False
        self._is_linalg_system = True

    def _find_non_zero_in_row(self, i: int, set_rows: list[bool] = None) -> list[int]:
        """Finds non-zero entries in row and returns indexes"""
        ret = []
        row = self._arr[i]
        for j in range(len(row)-1):
            if set_rows is not None:
                if set_rows[j]:
                    continue
            if row[j] != 0:
                ret.append(j)
        return ret

    def _find_non_zero_in_col(self, i: int, set_rows: list[bool] = None) -> list[int]:
        """Finds non-zero entries in col and returns indexes of row"""
        ret = []
        for j in range(self._row_num):
            if set_rows is not None:
                if set_rows[j]:
                    continue
            entry = self._arr[j][i]
            if entry != 0:
                ret.append(j)
        return ret

    def _has_only_one_non_zero_in_col(self, i: int, set_rows: list[bool] = None) -> bool:
        """checks if row/column at i had only one non-zero entry"""
        found_non_zero = False
        is_true = True
        for j in range(self._row_num):
            if set_rows is not None:
                if set_rows[j]:
                    continue
            entry = self._arr[j][i]
            if entry != 0:
                if found_non_zero:
                    is_true = False
                    break
                else:
                    found_non_zero = True

        return is_true

    @staticmethod
    def _is_all_set(arr: list[bool]) -> bool:
        """Checks if all bool are true in arr"""
        is_set = True
        for s in arr:
            if not s:
                is_set = False
        return is_set

    @staticmethod
    def update_dict(dic: dict[int, list[int]], rem: int):
        del dic[rem]
        for row, indexes in dic.items():
            if rem in indexes:
                dic[row].remove(rem)

    @staticmethod
    def swap_dict_keys(dic: dict[int, list[int]], i: int, j: int):
        temp = dic[i]
        dic[i] = dic[j]
        dic[j] = temp

    def optimise_for_solving(self) -> None:
        """swaps around rows based on zeroes in mat"""
        assert self._is_linalg_system, "Matrix is a linalg system"
        set_rows = [False for n in range(self._row_num)]
        n = self._row_num
        # set trivial rows and columns

        options_dict = {}
        for row in range(n):
            options_dict[row] = self._find_non_zero_in_row(row)

        while not self._is_all_set(set_rows):
            is_swap = False
            target = -1
            destination = -1
            for row, indexes in options_dict.items():
                if len(indexes) == 1:
                    destination, target, is_swap = indexes[0], row, True
                else:
                    found_swap = False
                    for e in indexes:
                        if self._has_only_one_non_zero_in_col(e, set_rows):
                            destination, target, is_swap = e, row, True
                            found_swap = True
                            break
                    if found_swap:
                        break
            if not is_swap:
                for row, indexes in options_dict.items():
                    if row in indexes:
                        self.update_dict(options_dict, row)
                        set_rows[row] = True
                        break
                    else:
                        indexes_c = self._find_non_zero_in_col(row, set_rows)
                        destination, target, is_swap = indexes_c[0], row, True
                        break

            if is_swap:
                self._row_swap(destination, target)
                self.swap_dict_keys(options_dict, destination, target)
                set_rows[destination] = True
                self.update_dict(options_dict, destination)

    def _row_add_row_mod(self, row1: int, row2: int, c: int, mod: int) -> None:
        """Add rows: row1 = row1+c*row2 % mod"""
        assert row1 < self._row_num and row2 < self._row_num, "Row number out of index"
        scaled_row2 = self._arr[row2]*c
        added_rows = np.add(self._arr[row1], scaled_row2)
        added_rows_mod = np.remainder(added_rows, mod)
        self._arr[row1] = added_rows_mod

    def _row_multiplication_mod(self, row: int, c: int, mod: int) -> None:
        """Multiply row by constant"""
        assert row < self._row_num, "Row number out of index"
        vec = np.array([1.0 for i in range(self._row_num)])
        vec[row] = c
        self._arr = np.remainder((self._arr.T * vec).T, mod)

    def solve_mod_p(self, mod: int) -> list[np.float64]:
        """Solve system mod n"""
        assert self._is_linalg_system, "Can only solve systems of linear equations"
        self._arr = np.remainder(self._arr, mod)
        self.optimise_for_solving()
        for k in range(self._row_num - 1):
            entry_value = self._arr[k][k]
            for i in range(k + 1, self._row_num):
                c = (self._arr[i][k] * self.find_mod_inverse(int(entry_value), mod)) % mod
                self._row_add_row_mod(i, k, -c, mod)

        for k in range(self._row_num - 1, -1, -1):
            val = self._arr[k][k]
            self._row_multiplication_mod(k, self.find_mod_inverse(int(val), mod), mod)
            for i in range(k - 1, -1, -1):
                val = self._arr[i][k]
                self._row_add_row_mod(i, k, -val, mod)

        return [row[self._col_num-1] for row in self._arr]

    @staticmethod
    def find_mod_inverse(x: int, mod: int) -> int:
        gcd, a, b = gcd_extended(x, mod)
        return a


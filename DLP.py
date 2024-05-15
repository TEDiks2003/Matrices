from Matrix import *
import numpy as np
from random import randint
from functools import reduce


def dlp(y: int, g: int, p: int, factors_of_n: list[int]) -> int:
    """
    Solves Discrete Log Problem y=g^x mod p

    @param y: y in ZZ_p
    @param g: g in ZZ_p
    @param p: p in NN
    @param factors_of_n: prime factorisation of p-1 e.g. [8, 9, 5, 11]
    @return: log_g (y) in ZZ_p
    """
    primes = [2, 3, 5, 7, 11, 13]
    assert p > 13, "Prime list needs to be adjusted as p<=13"

    # Generating system of linear equations
    while True:
        powers_of_g = []
        factors_of_powers_of_g = []
        i = randint(1, 300)
        checked = []
        while len(powers_of_g) < len(primes):
            x = (g**i) % p
            if x in checked:
                continue
            in_primes, factors = prime_factorise(x, primes)
            if in_primes:
                powers_of_g.append(i)
                factors_of_powers_of_g.append(factors)
            i = randint(1, 300)
        mat = Matrix(np.array(factors_of_powers_of_g))
        if mat.get_determinant != 0:
            valid = True
            for n in factors_of_n:
                if Matrix(np.remainder(mat.get_arr, n)).get_determinant == 0:
                    valid = False
            if valid:
                break

    print(primes)
    for i, f in zip(powers_of_g, factors_of_powers_of_g):
        print(f"{i}: {f}")

    mat.append_b(np.array(powers_of_g))

    # solving mod prime factors of p-1
    solutions = []
    for n in factors_of_n:
        mat_copy = Matrix(mat.get_arr.copy())
        pre_sol_mat = Matrix(mat_copy.get_arr.copy())
        pre_sol_mat.optimise_for_solving()
        sol = mat_copy.solve_mod_p(n)
        for i in range(len(sol)):
            summ = 0
            for j in range(len(sol)):
                summ += pre_sol_mat.get_arr[i][j]*sol[j]
            summ = summ % n
            if (pre_sol_mat.get_arr[i][len(sol)] % n)-summ != 0:
                print(f"n: {n}, i: {i}, b: {pre_sol_mat.get_arr[i][len(sol)] % n}, sum: {summ}")
                print(f"sol: {sol}")
                print(f" pre solve mat:\n {np.remainder(pre_sol_mat.get_arr, n)}")
        solutions.append(sol)

    solutions = np.array(solutions).T
    print(solutions)

    db = []
    for sol in solutions:
        db.append(crt(factors_of_n, sol))

    print(db)

    power = -1
    factors = []
    while power == -1:
        i = randint(1, 300)
        x = g**i % p
        in_primes, factors = prime_factorise(x, primes)
        if in_primes:
            power = i

    ret = 0
    for i in range(len(primes)):
        ret += db[i]*factors[i]
    ret -= power
    ret = ret % p
    return ret



def prime_factorise(x: int, s: list[int]) -> (bool, list[int] or None):
    """
    Find prime factorization in terms of p in S, if possible otherwise (False,None)
    @param x: integer to be factorised
    @param s: list of primes in S
    """
    ret = []
    for p in s:
        i = 0
        while x % p == 0:
            x = x/p
            i += 1
        ret.append(i)

    # if factorisation is complete
    if x == 1:
        return True, ret
    # if factorisation is not complete
    else:
        return False, None


def mod_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1


def crt(n, r):
    total = 0
    prod = reduce(lambda r, b: r*b, n)

    for n_i, r_i in zip(n, r):
        p = prod // n_i
        total += r_i * mod_inv(p, n_i) * p
    return total % prod

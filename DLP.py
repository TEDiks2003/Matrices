from Matrix import *
import numpy as np
from random import randint


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
            break

    print(primes)
    for i, f in zip(powers_of_g, factors_of_powers_of_g):
        print(f"{i}: {f}")

    mat.append_b(np.array(powers_of_g))

    # preparing system for solving
    print(mat)
    mat.optimise_for_solving()
    print(mat)

    # solving mod prime factors of p-1
    solutions = []
    for n in factors_of_n:
        mat_copy = Matrix(mat.get_arr.copy())
        sol = mat_copy.solve_mod_p(n)
        solutions.append(sol)

    print(solutions)




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

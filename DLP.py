
def dlp(y, g, p, s: dict[int, list[int]], p_min_1_factors: list[int]):
    """
    Solves Discrete Log Problem y=g^x mod p

    @param s: dictionary of powers of g as composition of powers of primes
    """
    primes = [2, 3 ,5 , 7, 11, 13]
    assert p>13, "Prime list needs to be adjusted as p<=13"

    b = list(s.keys())

    mat = list(s.items())

    sol_for_crt =[]
    for n in p_min_1_factors:
        sol_for_crt.append(solve_mat_mod_n(mat,b,n))

def solve_mat_mod_n(mat, b, n):
    ret = []
    return ret



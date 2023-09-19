import numpy as np
import time
from numba import njit
from spin_tools import tools

@njit(fastmath=True)
def fwht(u):
    v = u.copy()
    h = 1 
    lv = len(v)
    while h < lv:
        for i in range(0, lv, h * 2):
            for j in range(i, i + h):
                x = v[j]
                y = v[j + h]
                v[j] = x + y 
                v[j + h] = x - y 
        h *= 2
    return v

@njit(fastmath=True)
def fwht_roll(u, n):

    """
    Different implementation exploiting the fact that 
    the appropriate indices can be generated beforehand.

    This isn't actually faster than the explicit
    FWHT above but might be a useful starting point for 
    possible optimization or as an another way of looking 
    at the FWHT structure.
    """

    idx = np.arange(2**n)

    for i in range(n):

        i0 = np.array([bool((j//(2**i))%2) for j in idx])
        m0 = idx[~i0]
        m1 = idx[i0]
        
        a0 = u[m0]
        a1 = u[m1]

        u[m0] = a0 + a1
        u[m1] = a0 - a1

    return u

if __name__ == "__main__":

    # run tests if not used as module
    n = 18
    runs = 20
    perf = np.zeros((runs,2))

    for i in range(runs + 1):

        np.random.seed(0)
        u = np.random.uniform(-1,1,2**n)
        start = time.perf_counter()
        fwht(u)
        end = time.perf_counter()
        if i > 0:
            perf[i-1,0] = end - start

        np.random.seed(0)
        u = np.random.uniform(-1,1,2**n)
        start = time.perf_counter()
        fwht_roll(u, n)
        end = time.perf_counter()
        if i > 0:
            perf[i-1,1] = end - start

    print(np.mean(perf, axis = 0))
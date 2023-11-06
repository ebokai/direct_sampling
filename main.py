import numpy as np
from fwht import *
import time
import hgsbm

def generate_data(model, pars, n, states, N = 100000):

    """
    model:   list/array of integer values corresponding to interactions
    pars:    list/array of real valued parameters of same length as model
    n:       number of variables
    states:  list/array of all 2**n states in desired format
    N:       number of data samples to return
    """

    if len(model) != len(pars):
        raise ValueError('number of parameters should match number of interactions')

    g = np.zeros(2**n)
    g[model] = pars
    u = np.exp(fwht(g))
    p = u / np.sum(u)
    data = np.random.choice(states, N, p = p)

    return data

def generate_energy(model, pars, n, states, N = 100000):

    """
    model:   list/array of integer values corresponding to interactions
    pars:    list/array of real valued parameters of same length as model
    n:       number of variables
    states:  list/array of all 2**n states in desired format
    N:       number of data samples to return
    """

    if len(model) != len(pars):
        raise ValueError('number of parameters should match number of interactions')

    g = np.zeros(2**n)
    g[model] = pars
    u = np.exp(fwht(g))
    p = u / np.sum(u)
    idx = np.random.choice(np.arange(2**n), N, p = p)
    data = np.array(states)[idx]

    return data, u, idx


if __name__ == "__main__":

    q = 4
    m = 5
    n = q * m 
    beta = 0.99
    k = 3 
    c = 6
    mu = 0

    state_strings = [format(i,f'0{n}b') for i in range(2**n)]

    for i in range(1):

        start = time.perf_counter()

        edges = hgsbm.hgsbm(q, m, mu, c, k, prob = False)

        model = np.array([np.power(2,edge).sum() for edge in edges], dtype = int)

        pars = np.array([beta for edge in edges])

        data = generate_data(model, pars, n, state_strings)

        data, energy, idx = generate_energy(model, pars, n, state_strings)
        
        print(time.perf_counter() - start)

ui, ci = np.unique(idx, return_counts = True)

for u, c in zip(ui, ci):

    print(f'state: {state_strings[u]} with energy {np.log(energy[u]):.1f} occurs: {c} times')
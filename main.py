import numpy as np
from fwht import *

def generate_data(model, pars, n, states, N = 1000):

    """
    model: list of integer values corresponding to interactions
    pars: list of real valued parameters of same length as model
    n: number of variables
    states = list of all 2**n states in desired format
    N: number of data samples to return
    """

    if len(model) != len(pars):
        raise ValueError('number of parameters should match number of interactions')

    g = np.zeros(2**n)
    g[model] = pars
    u = np.exp(fwht(g))
    p = u / np.sum(u)
    data = np.random.choice(states, N, p = p)

    return data

if __name__ == "__main__":

    n = 10 
    state_strings = [format(i,f'0{n}b') for i in range(2**n)]

    model = np.random.choice(np.arange(1,2**n), 8, replace = False)
    pars = np.random.uniform(-1,1,len(model))

    data = generate_data(model, pars, n, state_strings)
    print(data)


# -*- coding: utf-8 -*-
"""
This code uses function CHQO_ND_Renyi() defined in AbInitioMC
for evaluation of Renyi entropy with metropolis algorithm. It measures
the kinetic energy of springs that are varied along the path in
thermodynamic integration.

Task is parallelised, and each step along the path is evaluated
using different processor. Number of steps is set by 2 * len(clex)

*************************************************************
Make sure that number of processors is compatible with clex.
*************************************************************

The code is called by running the command

python3 Example.py

in the terminal. The default parameters that I use are:

        NUMBER_OF_STEPS = 2 * 10 ** 9
        start = 10 ** 6
        SEED = 11
        CODE = 5
        class_move_number = 100
        SAMPLING_NUMBER = 1000
        order = 2
"""

import sys
import numpy as np
from AbInitioMC import CHQO_ND_Renyi
from timeit import default_timer as timer
from multiprocessing import Process
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

def main(args):
    beta = 1050
    design = 10.0
    NAME = 'lam6_doublew'
    index = 0
    LEN = 16
    clex = np.linspace(0, 1, LEN)
    m = 1836
    # lam and omega are used only if POTENTIAL_TYPE='analytic'
    lam = 1
    omega = 20

    procesi = []
    st = timer()
    for cle in clex:
        print('\n', cle)
        zeta = beta / min(max(6, int(beta / design)), 300)
        M = max(int(beta / zeta), 3)
        zeta = beta / M
        epsilon = np.sqrt(zeta / m)
        print('epsilon: ', epsilon, 'cle: ', cle)
        print('old one: ', np.sqrt(beta), 'cle: ', cle)

        # These are used only if POTENTIAL_TYPE='analytic'
        #===============================
        gamma = m * omega ** 2
        mass = [[m / 2, 0], [0, m / 2]]
        mu = [[0, gamma], [gamma, 0]] # mu is divided by 2 in the program
        lamda = [[lam, 0],[0, lam]]
        #===============================

        shift = [0, 0]
        partition1 = 0
        partition2 = 1
        cle1 = (1 - cle) ** 3
        cle2 = cle ** 3
        

        NUMBER_OF_STEPS = 2 * 10 ** 9
        start = 10 ** 6
        SEED = 11
        CODE = 5
        class_move_number = 100
        SAMPLING_NUMBER = 1000
        dimension=2
        order = 2
        procesi.append(Process(target=CHQO_ND_Renyi, args=(beta, zeta, cle1,
                            order, mass, mu, lamda, shift, partition1,
                            partition2, NUMBER_OF_STEPS, epsilon, start, SEED,
                            CODE, SAMPLING_NUMBER, dimension, True, False,
                            index, NAME, cle2, -10, 'realistic')))
        index += 1

    for p in procesi:
        p.start()
    for p in procesi:
        p.join()
    en = timer()
    print('TIME: ', en - st, '\n', flush=True)


if __name__ == "__main__":
    main(sys.argv)


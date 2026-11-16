# -*- coding: utf-8 -*-
"""
This code uses function run_renyi() defined in AbInitioMC
for evaluation of Renyi entropy with metropolis algorithm. It measures
the kinetic energy of springs that are varied along the path in
thermodynamic integration.

Task is parallelised, and each step along the path is evaluated
using different processor. Number of steps is set by 2 * len(path_points)

*************************************************************
Make sure that number of processors is compatible with path_points.
*************************************************************

The code is called by running the command

python3 Example.py

in the terminal. The default parameters that I use are:

        n_steps = 2 * 10 ** 9
        start = 10 ** 6
        random_seed = 11
        sampling_offset = 5
        reflection_interval = 100
        sampling_interval = 1000
        renyi_order = 2
"""

import sys
import numpy as np
from AbInitioMC import run_renyi
from timeit import default_timer as timer
from multiprocessing import Process
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

def main(args):
    beta = 1050
    design = 10.0
    name = 'lam6_doublew'
    saveto = './' # folder where the results are saved
    index = 0
    n_points = 16
    path_points = np.linspace(0, 1, n_points)
    m = 1836
    # quartic and omega are used only if potential_type='analytic'
    quartic = 1
    omega = 20

    processes = []
    start_time = timer()
    for lam in path_points:
        print('\n', lam)
        zeta = beta / min(max(6, int(beta / design)), 300)
        n_beads = max(int(beta / zeta), 3)
        zeta = beta / n_beads
        epsilon = np.sqrt(zeta / m)
        print('epsilon: ', epsilon, 'lam: ', lam)
        print('old one: ', np.sqrt(beta), 'lam: ', lam)

        # These are used only if potential_type='analytic'
        #===============================
        gamma = m * omega ** 2
        mass = [[m / 2, 0], [0, m / 2]]
        mu = [[0, gamma], [gamma, 0]] # mu is divided by 2 in the program
        lamda = [[quartic, 0],[0, quartic]]
        #===============================

        shift = [0, 0]
        partition1 = 0
        partition2 = 1
        lam1 = (1 - lam) ** 3
        lam2 = lam ** 3
        

        n_steps = 2 * 10 ** 9
        start = 10 ** 6
        random_seed = 11
        sampling_offset = 5
        reflection_interval = 100
        sampling_interval = 1000
        dimension=2
        renyi_order = 2 # only 2 is supported
        # which grid of potentials.npy (and row of distances.npy) is used
        real_index = 28
        processes.append(Process(target=run_renyi, args=(beta, zeta, lam1,
                            renyi_order, mass, mu, lamda, shift, partition1,
                            partition2, n_steps, epsilon, start, random_seed,
                            sampling_offset, sampling_interval, dimension,
                            True, False, index, name, lam2, real_index,
                            'realistic'),
                            kwargs={'saveto': saveto,
                                    'reflection_interval':
                                    reflection_interval}))
        index += 1

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end_time = timer()
    print('TIME: ', end_time - start_time, '\n', flush=True)


if __name__ == "__main__":
    main(sys.argv)


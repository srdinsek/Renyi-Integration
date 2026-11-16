# -*- coding: utf-8 -*-
"""
This code uses function cluster_monte_carlo defined in IsingMC_main_cluster
for evaluation of Renyi entropy with Wolf cluster algorithm. It measures
the kinetic energy of springs that are varied along the path in
thermodynamic integration.

Task is parallelised, and each step along the path is evaluated
using different processor. Number of steps is set by n_points.

*************************************************************
Make sure that number of processors is compatible with n_points.
*************************************************************

This code was written for the purpose of comparing the integration
method to the SWAP method at large N as a function of r - strength
of the magnetic field.

The code is called by running the command of the effect

python3 IsingMC_INT.py 3 0.2 64 0.0001 3.5 22 50000 full 16 ./

in the terminal. The 8-th parameter can be either 'half' or 'full'.
The last parameter is the folder where all results are saved.
The default parameters that I use are:

    start = 10 ** 4
    sampling_interval = 100
    sampling_offset = 0
    renyi_order = 2
    n_walkers = 5

"""


import numpy as np
import os
import sys
from timeit import default_timer as timer
from multiprocessing import Process
import multiprocessing as mp

from IsingMC_main_cluster import run_integral

print("Number of processors: ", mp.cpu_count())

def main(args):
    print(args[0], flush=True)
    if len(args) == 10:
       raise ValueError("saveto is missing: give the folder where results "
                        "are saved as the last argument")
    if len(args) != 11:
       print("10 arguments required:\n" \
                + "betaJ, zeta, n_spins = args[1], args[2], args[3]\n"\
                + "r_min, r_max, n_fields = args[4], args[5], args[6]\n"\
                + "n_steps = args[7]\n"\
                + "subsystem = args[8]\n"\
                + "n_points (num. of CPU) = args[9]\n"\
                + "saveto = args[10]\n")
       return None

    #-------------------------------------------
    # Parameters
    #-------------------------------------------

    betaJ, zeta, n_spins = float(args[1]), float(args[2]), int(args[3])
    r_min, r_max = float(args[4]), float(args[5])
    n_fields, n_steps = int(args[6]), int(args[7])
    subsystem = args[8] # either 'half' or 'full'

    start = 10 ** 4
    sampling_interval = 100
    print('sampling number: ', sampling_interval)
    sampling_offset = 0
    renyi_order = 2 # Must be set to 2
    n_walkers = 5


    n_points = int(args[9])
    saveto = args[10]
    r_values = np.linspace(r_min, r_max, n_fields)
    # weights defining the path
    path_points = np.linspace(0, np.pi / 2, n_points)
    lam0_points = np.cos(path_points) ** 12
    lam1_points = np.sin(path_points) ** 12

    #-------------------------------------------
    # Algorithm
    #-------------------------------------------
    T0, T1 = [], []
    error0, error1 = [], []
    # loop over the strenghts of the magnetic field
    for r in r_values:
       T0_points, T1_points = np.zeros(n_points), np.zeros(n_points)
       error0_points = np.zeros(n_points)
       error1_points = np.zeros(n_points)

       print('r: ', r, flush=True)
       processes = []
       queues = []
       results = []

       #-------------------------------------------
       # Loop over the processes
       #-------------------------------------------
       start_time = timer()
       for i in range(n_points):
          # bag into which we will save results
          queues.append(mp.Queue())
          if subsystem == 'half':
             # list of processes
             processes.append(Process(target=run_integral, args=(betaJ,
                      zeta, n_spins, r, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, n_spins // 2, 0,
                      n_walkers, lam0_points[i], lam1_points[i],
                      queues[-1], i, saveto)))
          else:
             # list of processes
             processes.append(Process(target=run_integral, args=(betaJ,
                      zeta, n_spins, r, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, n_spins, 0,
                      n_walkers, lam0_points[i], lam1_points[i],
                      queues[-1], i, saveto)))
       for p in processes:
          # start the process p
          p.start()
       for p in processes:
          p.join()
       end_time = timer()
       print('TIME: ', end_time - start_time, '\n', flush=True)

       for p in queues:
          # move all the results from the bag to the list
          results.append(p.get())

       #-------------------------------------------
       # Post processing of results
       #-------------------------------------------
       # because processes don't necessarily start and end in the same order,
       # we recognise them by index.
       for (index, lam0, lam1, T0_walkers, T1_walkers) in results:

          # kinetic energy of the two springs forming split ensemble
          t0 = np.sum(T0_walkers) / len(T0_walkers)
          # kinetic energy of the two springs forming joint ensemble
          t1 = np.sum(T1_walkers) / len(T1_walkers)

          # error-bars of energies: the spread (standard deviation) of
          # the walker averages, not the standard error of their mean
          err0 = np.sqrt(np.sum([(t0 - T0_walkers[i]) ** 2
                                     for i in range(n_walkers)]) / n_walkers)
          err1 = np.sqrt(np.sum([(t1 - T1_walkers[i]) ** 2
                                     for i in range(n_walkers)]) / n_walkers)

          # log2 is not used here, instead log is used already in the
          # definition of the observable.
          T0_points[index] += t0
          T1_points[index] += t1
          error0_points[index] += err0
          error1_points[index] += err1

       T0.append(T0_points)
       T1.append(T1_points)
       error0.append(error0_points)
       error1.append(error1_points)

    T0 = np.array(T0)
    T1 = np.array(T1)
    error0 = np.array(error0)
    error1 = np.array(error1)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    tag = ('_fast_BetaJ_' + str(betaJ) + '_zeta_' + str(zeta) + '_N_'
           + str(n_spins) + '_r_dep' + str(n_points))
    parameters = np.array([betaJ, zeta, n_spins, r_min, r_max, n_fields,
                           n_steps, sampling_interval, start, sampling_offset])
    for name, data in [('pCLiT0', T0), ('pCLiT1', T1),
                       ('pCLiERROR_T0', error0), ('pCLiERROR_T1', error1),
                       ('pCLiargs_T1', parameters)]:
        np.save(os.path.join(saveto, subsystem + name + tag), data)

if __name__ == "__main__":
    main(sys.argv)

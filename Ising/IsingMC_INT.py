# -*- coding: utf-8 -*-
"""
This code uses function MontePath_Master defined in IsingMC_main_cluster
for evaluation of Renyi entropy with Wolf cluster algorithm. It measures
the kinetic energy of springs that are varied along the path in
thermodynamic integration.

Task is parallelised, and each step along the path is evaluated
using different processor. Number of steps is set by LEN.

*************************************************************
Make sure that number of processors is compatible with LEN.
*************************************************************

This code was written for the purpose of comparing the integration
method to the SWAP method at large N as a function of r - strength
of the magnetic field.

The code is called by running the command of the effect

python3 IsingMC_INT.py 3 0.2 64 0.0001 3.5 22 50000 full 16

in the terminal. The 8-th parameter can be either 'half' or 'full'.
The default parameters that I use are:

    start = 10 ** 4
    SAMPLING_NUMBER = 100
    CODE = 0
    RENYI_ORDER = 2
    N_WALKERS = 5
    LONG = (60 * NUMBER_OF_STEPS - start) // SAMPLING_NUMBER

"""


import numpy as np
import os
import sys
from timeit import default_timer as timer
from multiprocessing import Process
import multiprocessing as mp

from IsingMC_main_cluster import MontePath_Master
from IsingMC_main_cluster import MAIN_integral, MAIN_ratio

print("Number of processors: ", mp.cpu_count())
SAVETO = './'

def main(args):
    print(args[0], flush=True)
    if len(args) != 10:
       print("8 arguments required:\n" \
                + "betaJ, zeta, N = args[1], args[2], args[3]\n"\
                + "r_min, r_max, N_steps = args[4], args[5], args[6]\n"\
                + "NUMBER_OF_STEPS = args[7]\n"\
                + "GESLO = args[8]"\
                + "LEN (num. of CPU) = args[9]\n")
       return None

    #-------------------------------------------
    # Parameters
    #-------------------------------------------

    betaJ, zeta, N = float(args[1]), float(args[2]), int(args[3])
    r_min, r_max = float(args[4]), float(args[5])
    N_steps, NUMBER_OF_STEPS = int(args[6]), int(args[7])
    GESLO = args[8] # either 'half' or 'full'

    start = 10 ** 4
    SAMPLING_NUMBER = 100
    print('sampling number: ', SAMPLING_NUMBER)
    CODE = 0
    RENYI_ORDER = 2 # Must be set to 2
    N_WALKERS = 5
    LONG = (60 * NUMBER_OF_STEPS - start) // SAMPLING_NUMBER


    LEN = int(args[9])
    rr = np.linspace(r_min, r_max, N_steps)
    # weights defining the path
    clex = np.linspace(0, np.pi / 2, LEN)
    cle1 = np.cos(clex) ** 12
    cle2 = np.sin(clex) ** 12

    #-------------------------------------------
    # Algorithm
    #-------------------------------------------
    T0, T1 = [], []
    ERR0, ERR1 = [], []
    # loop over the strenghts of the magnetic field
    for r in rr:
       GATHER_T0, GATHER_T1 = np.zeros(LEN), np.zeros(LEN)
       GATHER_ERROR0, GATHER_ERROR1 = np.zeros(LEN), np.zeros(LEN)

       print('r: ', r, flush=True)
       procesi = []
       queueti = []
       ruru_all = []

       #-------------------------------------------
       # Loop over the processes
       #-------------------------------------------
       st = timer()
       for i in range(LEN):
          # bag into which we will save results
          queueti.append(mp.Queue())
          if GESLO == 'half':
             # list of processes
             procesi.append(Process(target=MAIN_integral, args=(betaJ, zeta,
                      N, r, NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, N // 2, 0, N_WALKERS, cle1[i], cle2[i],
                      queueti[-1], i)))
          else:
             # list of processes
             procesi.append(Process(target=MAIN_integral, args=(betaJ, zeta,
                      N, r, NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, N, 0, N_WALKERS, cle1[i], cle2[i],
                      queueti[-1], i)))
       for p in procesi:
          # start the process p
          p.start()
       for p in procesi:
          p.join()
       en = timer()
       print('TIME: ', en - st, '\n', flush=True)

       for p in queueti:
          # move all the results from the bag to the list
          ruru_all.append(p.get())

       #-------------------------------------------
       # Post processing of results
       #-------------------------------------------
       # because processes don't necessarily start and end in the same order,
       # we recognise them by index.
       for (index, c0, c1, ruru0, ruru1) in ruru_all:

          # kinetic energy of the two springs forming split ensemble
          t0 = np.sum(ruru0) / len(ruru0)
          # kinetic energy of the two springs forming joint ensemble
          t1 = np.sum(ruru1) / len(ruru1)

          # error-bars of energies, based on the variance of the walkers
          err0 = np.sqrt(np.sum([(t0 - ruru0[i]) ** 2
                                     for i in range(N_WALKERS)]) / N_WALKERS)
          err1 = np.sqrt(np.sum([(t1 - ruru1[i]) ** 2
                                     for i in range(N_WALKERS)]) / N_WALKERS)

          # log2 is not used here, instead log is used already in the
          # definition of the observable.
          GATHER_T0[index] += t0
          GATHER_T1[index] += t1
          GATHER_ERROR0[index] += err0
          GATHER_ERROR1[index] += err1

       T0.append(GATHER_T0)
       T1.append(GATHER_T1)
       ERR0.append(GATHER_ERROR0)
       ERR1.append(GATHER_ERROR1)

    T0 = np.array(T0)
    T1 = np.array(T1)
    ERR0 = np.array(ERR0)
    ERR1 = np.array(ERR1)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    np.save(os.path.join(SAVETO, GESLO + 'pCLiT0_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            T0)
    np.save(os.path.join(SAVETO, GESLO + 'pCLiT1_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            T1)
    np.save(os.path.join(SAVETO, GESLO + 'pCLiERROR_T0_fast_BetaJ_'\
            + str(betaJ) + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep'\
            + str(LEN)),
            ERR0)
    np.save(os.path.join(SAVETO, GESLO + 'pCLiERROR_T1_fast_BetaJ_'\
            + str(betaJ) + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep'\
            + str(LEN)),
            ERR1)
    np.save(os.path.join(SAVETO, GESLO + 'pCLiargs_T1_fast_BetaJ_'\
            + str(betaJ) + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep'\
            + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps,
            NUMBER_OF_STEPS, SAMPLING_NUMBER, start, CODE]))
    np.save(os.path.join(SAVETO, GESLO + 'pCLiargs_T1_fast_BetaJ_'\
            + str(betaJ) + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep'\
            + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps,
            NUMBER_OF_STEPS, SAMPLING_NUMBER, start, CODE]))

if __name__ == "__main__":
    main(sys.argv)

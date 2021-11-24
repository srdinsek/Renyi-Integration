"""
This coed uses function MontePath_Master defined in IsingMC_main_cluster
for evaluation of Renyi entropy with Wolf cluster algorithm. It measures
the SWAP operator and uses the trick

Z_n / Z_0 = Z_n / Z_(n-1) * Z_(n-1) / Z_(n-2) * ... * Z_2 / Z_1 * Z_1 / Z_0

Number of such intermediate steps is set by LEN.

*************************************************************
ATTENTION! - To get the full entropy N // LEN == int(N / LEN)
*************************************************************

Make sure that number of processors is compatible with LEN.


This code was written for the pourpose of comparing the integration
method to the SWAP method at large N as a funcction of r - strength
of the magnetic field.

The code is called by running the command of the effect

python3 IsingMC_RAT.py 3 0.2 64 0.0001 3.5 22 50000 8

in the terminal. The default parameters that I use are:

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
    if len(args) != 9:
       print("8 arguments required:\n" \
                + "betaJ, zeta, N = args[1], args[2], args[3]\n"\
                + "r_min, r_max, N_steps = args[4], args[5], args[6]\n"\
                + "NUMBER_OF_STEPS = args[7]"\
                + "LEN(num. of CPU) = args[8]\n")
       return None
    
    #-------------------------------------------
    # Parameters
    #-------------------------------------------

    betaJ, zeta, N = float(args[1]), float(args[2]), int(args[3])
    r_min, r_max = float(args[4]), float(args[5])
    N_steps, NUMBER_OF_STEPS = int(args[6]), int(args[7])

    start = 10 ** 4
    SAMPLING_NUMBER = 100
    print('sampling number: ', SAMPLING_NUMBER)
    CODE = 0
    RENYI_ORDER = 2 # Must be set to 2
    N_WALKERS = 5
    LONG = (60 * NUMBER_OF_STEPS - start) // SAMPLING_NUMBER

    LEN = int(args[8])
    rr = np.linspace(r_min, r_max, N_steps)

    #-------------------------------------------
    # Algorithm
    #-------------------------------------------
    RENYI1, RENYI2, RENYI_TWO, RENYI = [], [], [], []
    ERR1, ERR2, ERR = [], [], []
    # loop over the strenghts of the magnetic field
    for r in rr:
       GATHER_RENYI1 = np.zeros(LEN)
       GATHER_RENYI2 = np.zeros(LEN)
       GATHER_RENYI = np.zeros(LEN)
       GATHER_RENYI_TWO = np.zeros(LEN)
       GATHER_ERROR = np.zeros(LEN)
       GATHER_ERROR1 = np.zeros(LEN)
       GATHER_ERROR2 = np.zeros(LEN)

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
          # list of processes
          procesi.append(Process(target=MAIN_ratio, args=(betaJ, zeta, N, r,
                      NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, (N // LEN) * (i + 1), (N // LEN) * i,
                      N_WALKERS, queueti[-1], i)))
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
       # we recognise them by ip.
       for (ip, ru1, ru2, ruru1, ruru2) in ruru_all:
          index = ip // (N // LEN)

          # different SWAP opperators
          ratio1 = np.sum(ruru1) / len(ruru1)
          ratio2 = np.sum(ruru2) / len(ruru2)
          ratio_two = np.sum(ru2) / np.sum(ru1)
          ratio = ratio2 / ratio1

          # entropy form different SWAP opperators
          entropy1 = - np.log2(ratio1)
          entropy2 = np.log2(ratio2)
          entropy = np.log2(ratio) / 2
          entropy_two = np.log2(ratio_two)

          # errors for different SWAP opperators, based on the variance
          # of the walkers
          err1 = np.sqrt(np.sum([(ratio1 - ruru1[i]) ** 2
                                     for i in range(N_WALKERS)]) / N_WALKERS)
          err2 = np.sqrt(np.sum([(ratio2 - ruru2[i]) ** 2
                                     for i in range(N_WALKERS)]) / N_WALKERS)

          err = err1 / (np.sum(ruru2) + err2)\
                + err2 * (np.sum(ruru1) + err1) / (np.sum(ruru2) + err2) ** 2

          # Attention! - log2 is used!
          err = err / ratio / np.log2(np.e)
          err1 = err1 / ratio1 / np.log2(np.e)
          err2 = err2 / ratio2 / np.log2(np.e)

          GATHER_RENYI1[index] += entropy1
          GATHER_RENYI2[index] += entropy2
          GATHER_RENYI[index] += entropy
          GATHER_RENYI_TWO[index] += entropy_two
          GATHER_ERROR[index] += err
          GATHER_ERROR1[index] += err1
          GATHER_ERROR2[index] += err2

       RENYI.append(GATHER_RENYI)
       RENYI1.append(GATHER_RENYI1)
       RENYI2.append(GATHER_RENYI2)
       RENYI_TWO.append(GATHER_RENYI_TWO)
       ERR.append(GATHER_ERROR)
       ERR1.append(GATHER_ERROR1)
       ERR2.append(GATHER_ERROR2)

    RENYI = np.array(RENYI)
    RENYI1 = np.array(RENYI1)
    RENYI2 = np.array(RENYI2)
    RENYI_TWO = np.array(RENYI_TWO)
    ERR = np.array(ERR)
    ERR1 = np.array(ERR1)
    ERR2 = np.array(ERR2)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    np.save(os.path.join(SAVETO, 'pCLrRTW_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            RENYI)
    np.save(os.path.join(SAVETO, 'pCLrERROR_RTW_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            ERR)
    np.save(os.path.join(SAVETO, 'pCLrargs_RTW_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps, NUMBER_OF_STEPS,
            SAMPLING_NUMBER, start, CODE]))
    np.save(os.path.join(SAVETO, 'pCLrRTW1_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            RENYI1)
    np.save(os.path.join(SAVETO, 'pCLrERROR_RTW1_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            ERR1)
    np.save(os.path.join(SAVETO, 'pCLrargs_RTW1_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps, NUMBER_OF_STEPS,
            SAMPLING_NUMBER, start, CODE]))
    np.save(os.path.join(SAVETO, 'pCLrRTW2_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            RENYI2)
    np.save(os.path.join(SAVETO, 'pCLrERROR_RTW2_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            ERR2)
    np.save(os.path.join(SAVETO, 'pCLrargs_RTW2_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps, NUMBER_OF_STEPS,
            SAMPLING_NUMBER, start, CODE]))
    np.save(os.path.join(SAVETO, 'pCLrRTW_two_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            RENYI_TWO)
    np.save(os.path.join(SAVETO, 'pCLrargs_RTW_two_fast_BetaJ_' + str(betaJ)\
            + '_zeta_' + str(zeta) + '_N_' + str(N) + '_r_dep' + str(LEN)),
            np.array([betaJ, zeta, N, r_min, r_max, N_steps, NUMBER_OF_STEPS,
            SAMPLING_NUMBER, start, CODE]))

if __name__ == "__main__":
    main(sys.argv)

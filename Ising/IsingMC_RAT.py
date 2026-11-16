"""
This coed uses function cluster_monte_carlo defined in IsingMC_main_cluster
for evaluation of Renyi entropy with Wolf cluster algorithm. It measures
the SWAP operator and uses the trick

Z_n / Z_0 = Z_n / Z_(n-1) * Z_(n-1) / Z_(n-2) * ... * Z_2 / Z_1 * Z_1 / Z_0

Number of such intermediate steps is set by n_points.

*************************************************************
ATTENTION! - To get the full entropy
  n_spins // n_points == int(n_spins / n_points)
*************************************************************

Make sure that number of processors is compatible with n_points.


This code was written for the pourpose of comparing the integration
method to the SWAP method at large n_spins as a funcction of r - strength
of the magnetic field.

The code is called by running the command of the effect

python3 IsingMC_RAT.py 3 0.2 64 0.0001 3.5 22 50000 8 ./

in the terminal. The last parameter is the folder where all results
are saved. The default parameters that I use are:

    start = 10 ** 4
    sampling_interval = 100
    sampling_offset = 0
    renyi_order = 2
    n_walkers = 5
    LONG = (60 * n_steps - start) // sampling_interval

"""


import numpy as np
import os
import sys
from timeit import default_timer as timer
from multiprocessing import Process
import multiprocessing as mp

from IsingMC_main_cluster import cluster_monte_carlo
from IsingMC_main_cluster import run_integral, run_ratio

print("Number of processors: ", mp.cpu_count())

def main(args):
    print(args[0], flush=True)
    if len(args) == 9:
       raise ValueError("saveto is missing: give the folder where results "
                        "are saved as the last argument")
    if len(args) != 10:
       print("9 arguments required:\n" \
                + "betaJ, zeta, n_spins = args[1], args[2], args[3]\n"\
                + "r_min, r_max, n_fields = args[4], args[5], args[6]\n"\
                + "n_steps = args[7]\n"\
                + "n_points (num. of CPU) = args[8]\n"\
                + "saveto = args[9]\n")
       return None
    
    #-------------------------------------------
    # Parameters
    #-------------------------------------------

    betaJ, zeta, n_spins = float(args[1]), float(args[2]), int(args[3])
    r_min, r_max = float(args[4]), float(args[5])
    n_fields, n_steps = int(args[6]), int(args[7])

    start = 10 ** 4
    sampling_interval = 100
    print('sampling number: ', sampling_interval)
    sampling_offset = 0
    renyi_order = 2 # Must be set to 2
    n_walkers = 5
    LONG = (60 * n_steps - start) // sampling_interval

    n_points = int(args[8])
    saveto = args[9]
    r_values = np.linspace(r_min, r_max, n_fields)

    #-------------------------------------------
    # Algorithm
    #-------------------------------------------
    renyi1, renyi2, renyi_two, renyi = [], [], [], []
    error1, error2, error = [], [], []
    # loop over the strenghts of the magnetic field
    for r in r_values:
       renyi1_points = np.zeros(n_points)
       renyi2_points = np.zeros(n_points)
       renyi_points = np.zeros(n_points)
       renyi_two_points = np.zeros(n_points)
       error_points = np.zeros(n_points)
       error1_points = np.zeros(n_points)
       error2_points = np.zeros(n_points)

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
          # list of processes
          processes.append(Process(target=run_ratio, args=(betaJ, zeta,
                      n_spins, r, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order,
                      (n_spins // n_points) * (i + 1),
                      (n_spins // n_points) * i,
                      n_walkers, queues[-1], i, saveto)))
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
       # we recognise them by partition1.
       for (partition1, swap_min1, swap_min2, swap1, swap2) in results:
          index = partition1 // (n_spins // n_points)

          # different SWAP opperators
          ratio1 = np.sum(swap1) / len(swap1)
          ratio2 = np.sum(swap2) / len(swap2)
          ratio_two = np.sum(swap_min2) / np.sum(swap_min1)
          ratio = ratio2 / ratio1

          # entropy form different SWAP opperators
          entropy1 = - np.log2(ratio1)
          entropy2 = np.log2(ratio2)
          entropy = np.log2(ratio) / 2
          entropy_two = np.log2(ratio_two)

          # errors for different SWAP opperators, based on the variance
          # of the walkers
          err1 = np.sqrt(np.sum([(ratio1 - swap1[i]) ** 2
                                     for i in range(n_walkers)]) / n_walkers)
          err2 = np.sqrt(np.sum([(ratio2 - swap2[i]) ** 2
                                     for i in range(n_walkers)]) / n_walkers)

          err = err1 / (np.sum(swap2) + err2)\
                + err2 * (np.sum(swap1) + err1) / (np.sum(swap2) + err2) ** 2

          # Attention! - log2 is used!
          err = err / ratio / np.log2(np.e)
          err1 = err1 / ratio1 / np.log2(np.e)
          err2 = err2 / ratio2 / np.log2(np.e)

          renyi1_points[index] += entropy1
          renyi2_points[index] += entropy2
          renyi_points[index] += entropy
          renyi_two_points[index] += entropy_two
          error_points[index] += err
          error1_points[index] += err1
          error2_points[index] += err2

       renyi.append(renyi_points)
       renyi1.append(renyi1_points)
       renyi2.append(renyi2_points)
       renyi_two.append(renyi_two_points)
       error.append(error_points)
       error1.append(error1_points)
       error2.append(error2_points)

    renyi = np.array(renyi)
    renyi1 = np.array(renyi1)
    renyi2 = np.array(renyi2)
    renyi_two = np.array(renyi_two)
    error = np.array(error)
    error1 = np.array(error1)
    error2 = np.array(error2)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    tag = ('_fast_BetaJ_' + str(betaJ) + '_zeta_' + str(zeta) + '_N_'
           + str(n_spins) + '_r_dep' + str(n_points))
    parameters = np.array([betaJ, zeta, n_spins, r_min, r_max, n_fields,
                           n_steps, sampling_interval, start, sampling_offset])
    for name, data in [('pCLrRTW', renyi), ('pCLrERROR_RTW', error),
                       ('pCLrRTW1', renyi1), ('pCLrERROR_RTW1', error1),
                       ('pCLrRTW2', renyi2), ('pCLrERROR_RTW2', error2),
                       ('pCLrRTW_two', renyi_two)]:
        np.save(os.path.join(saveto, name + tag), data)
    for name in ['pCLrargs_RTW', 'pCLrargs_RTW1', 'pCLrargs_RTW2',
                 'pCLrargs_RTW_two']:
        np.save(os.path.join(saveto, name + tag), parameters)

if __name__ == "__main__":
    main(sys.argv)

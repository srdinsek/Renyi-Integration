# -*- coding: utf-8 -*-
"""
This is Wolf cluster algorithm adapted for the computation of
entanglement Renyi entropy in 1D Quantum Ising Model. It is 
primarelly meant for the calculation of the second order 
Renyi entropy, but it can be easily adapted for higher orders.

Main function MontePath_Master, can be used for any (connected)
partition size and thermodynamic integration. Thus also for
SWAP operator.

For the two observables investigated in the paper, we use
rutines:
1) IsingMC_RAT.py  (for evaluation by SWAP)
2) IsingMC_INT.py  (for evaluation by thermodynamic integration)
each of them running the same algorithm, but for different set of
parameters and observables.

This code was written for the pourpose of comparing the integration
method to the SWAP method at large N.

The main algorithm strongly depends on subrutines defined in 
IsingMC_main_lib. It's performance is enhaced by the extensive
use of numba's decorator @njit.
"""

import numpy as np
from numba import jit, njit
from numpy.random import rand,randint,normal,uniform,seed
import math

from IsingMC_main_lib import EXPAND,observable,KineticAB

SAVETO = '/scratchalpha/srdinsek/IsingModel/Ising_1D_Renyi/'
# define the directory where results of 1) and 2) will be saved


@njit(error_model="numpy")
def MontePath_Master(betaJ, zeta, N, r, InitialState, IS, NUMBER_OF_STEPS,
                start, SAMPLING_NUMBER, CODE, RENYI_ORDER, partition2,
                partition1, N_WALKERS, cle0, cle1, TYPE_OF_OBS_STR):
    '''The algorithm was written with H = s^x_i s^x_{i+1} + r s^z_i  
    in mind. Summ is over the i with periodic boundary conditions.
    Parameter ´partition1´ specifies the number of joint copies
    starting from i=0. Zero means there are none. Parameter
    ´partition2´ specifies the trial size starting from i=0. If
    ´partition1´ is nonzero, then the trial size is 
    partition2 - partition1. Trial copies are the ones that we
    propose to SWAP or that we modify during the thermodynamic
    integration. For example function with parameters partition2,
    partition1=0 simulates split ensemble and modifies the
    subsystem spaning from i=0 to i=(partition2 - 1). The
    function with parameters partition2=0, partition1 simulates 
    ensemlbe, where copies spanning from i=0 to i=(partition1 - 1)
    are joined.

    Besides the cluster algorithm, the code uses also the
    multiwalkers algorithm. Number of walkers is set by N_WALKERS.
    '''
    # betaJ = inverse temperature
    # zeta = betaJ / N_beads
    # N = number of particles
    # r = strenght of the magnetic field
    # InitialState = initial state (used only if IS != 0)
    # IS = seed, but random state is used only if IS = 0
    # NUMBER_OF_STEPS = number of clusters produced
    # start = number of steps when sampling starts
    # SAMPLING_NUMBER = frequency of sampling
    # CODE = sampling phase shift. Usually set to 0
    # RENYI_ORDER = order of Renyi entropy. Only 2 can be used
    # N_WALKERS = number of independent walkers.
    # cle0 = weight on the springs that fulfil the split copies
    # boundary condition
    # cle1 = weight on the sping joining two copies.
    # TYPE_OF_OBS_STR = either 'integral' or 'ratio' - specifies
    # the observables that we want to sample - energy or SWAP.

    if TYPE_OF_OBS_STR == 'integral':

        TYPE_OF_OBS = 0

    if TYPE_OF_OBS_STR == 'ratio':

        TYPE_OF_OBS = 1

    # Because clusters are biger at small r, we
    # make less moves
    if r > 3:
        NUMBER_OF_STEPS = NUMBER_OF_STEPS * 60
    elif r > 1:
        NUMBER_OF_STEPS = NUMBER_OF_STEPS * 10
    elif r < 1:
        SAMPLING_NUMBER = 10

    # zeta is redefined, so that M is indeed an integer
    M = int(betaJ / zeta)
    print('M : ', M)
    zeta = betaJ / M

    # Constants used in cluster algorithm
    ch, sh = np.cosh(betaJ * r / M), np.sinh(betaJ * r / M)
    const = np.log(1 / np.tanh(betaJ * r / M)) / 2
    pK = 1 - np.exp(-2 * const) # link in imaginary-time dir. - "kinetic"
    pP = 1 - np.exp(-2 * betaJ / M) # link along the bead - "potential"
    pK1 = 1 - np.exp(-2 * const * cle0) # link1 in imag.-time at the boundary
    pK2 = 1 - np.exp(-2 * const * cle1) # link2 in imag.-time at the boundary
    TIME = 2 * M # assumed RENYI_ORDER = 2
    FULLSIZE = 2 * M * N # assumed RENYI_ORDER = 2
    print(ch, sh, const)

    # initial state for each walker
    if IS > 0:
        seed(IS)
        state = InitialState.astype(np.bool_)
    else:
        seed(0)
        state = randint(0, 2, (N_WALKERS, N, 2 * M)).astype(np.bool_)

    # counts number of sample points for each walker
    n_used = np.zeros(N_WALKERS, dtype=np.int64)

    # averages of observables
    exp = np.zeros((N_WALKERS), dtype=np.int64) # number of accepted moves
    T0_avg = np.zeros((N_WALKERS), dtype=np.float64)
    T1_avg = np.zeros((N_WALKERS), dtype=np.float64)
    Ratio = np.zeros((N_WALKERS), dtype=np.float64) #SWAP
    Ratio_two = np.zeros((N_WALKERS), dtype=np.float64)

    # full observables
    T0_all = np.zeros((N_WALKERS,
          (NUMBER_OF_STEPS - start) // SAMPLING_NUMBER), dtype=np.float64)
    T1_all = np.zeros((N_WALKERS,
          (NUMBER_OF_STEPS - start) // SAMPLING_NUMBER), dtype=np.float64)
    Ratio_all = np.zeros((N_WALKERS,
          (NUMBER_OF_STEPS - start) // SAMPLING_NUMBER), dtype=np.float64)
    Ratio_two_all = np.zeros((N_WALKERS,
          (NUMBER_OF_STEPS - start) // SAMPLING_NUMBER), dtype=np.float64)

    # some arrays that help reducing memorry used
    q = np.zeros((2 * M), dtype=np.bool_)
    CLUSTER_indi = np.zeros((N_WALKERS, FULLSIZE, 2), dtype=np.int64)
    CLUSTER_name = np.zeros((N_WALKERS, FULLSIZE), dtype=np.int64)
    POCKET = np.zeros((N_WALKERS, FULLSIZE), dtype=np.int64)

    # Parameters used for the cluster algortihm
    DE = 0
    AVSIZ = 0

    # We start the Metropolis loop
    for step in range(NUMBER_OF_STEPS):
        # Each walker is treated separately
        for walker in range(N_WALKERS):

            ''' We construct the new move. '''
            #-------------------------------------------
            # Wolf cluster move
            #-------------------------------------------

            # randomly choosen starting spin at position (n, m)
            n = randint(0, N)
            m = randint(0, TIME)
            
            # empty the array
            CLUSTER_indi.fill(0)

            # add the spin (n, m) to the cluster
            CLUSTER_indi[walker, 0, 0] = n
            CLUSTER_indi[walker, 0, 1] = m

            # add the spin (n, m) to the pocket in number format
            POCKET[walker, 0] = m + n * TIME
            
            # save the position in a number format
            CLUSTER_name[walker, 0] = m + n * TIME

            # size of the pocket
            SIZE = 1
            # size of the cluster
            SIZE_C = 1

            # loop over the pocket until it is empty
            while SIZE > 0:
                # read the position of the last spin in the pocket
                n = POCKET[walker, SIZE - 1] // TIME
                m = POCKET[walker, SIZE - 1] % TIME

                # Check the neighbors of the spin from the pocket
                # and add them to the cluster with the physical
                # probabilities with the function EXPAND()
                SIZE_C, POCKET[walker], SIZE,\
                CLUSTER_indi[walker], CLUSTER_name[walker] = EXPAND(N, M,
                              TIME, const, pP, pK, pK1, pK2, partition1,
                              partition2, state[walker], POCKET[walker],
                              CLUSTER_indi[walker], SIZE, SIZE_C, n, m,
                              CLUSTER_name[walker])
                # The used spin is removed from the pocket, but
                # the new ones are added. They are added also
                # to CLUSTER_indi

            # Flip the cluster
            # state[walker] = np.abs(state[walker] - CLUSTER[walker]) # slow
            for i in range(SIZE_C):
                n, m = CLUSTER_indi[walker, i, 0], CLUSTER_indi[walker, i, 1]
                state[walker, n, m] = not state[walker, n, m]
            # every step is accepted
            exp[walker] += 1
            AVSIZ += SIZE_C

            ''' Here we meassure our observable. '''
            if step % SAMPLING_NUMBER == CODE and step > start:
                if TYPE_OF_OBS == 0:

                    # we record the energy of the spings along the path
                    T0, T1 = observable(N, M, const,
                                    partition1, partition2, state[walker])
                    T0_avg[walker] += T0
                    T1_avg[walker] += T1
                    T0_all[walker, n_used[walker]] += T0
                    T1_all[walker, n_used[walker]] += T1

                    n_used[walker] += 1

                if TYPE_OF_OBS == 1:

                    # we evaluate the SWAP operator
                    if cle0 == 0:
                        Kq_new = KineticAB(N, M, ch, sh, partition1,
                                            partition2, state[walker])
                        DGRANDE = KineticAB(N, M, ch, sh,
                                            partition2, partition1,
                                                state[walker]) / Kq_new
                        DGRANDE = 1 / DGRANDE
                    else:
                        Kq_new = KineticAB(N, M, ch, sh, partition1,
                                           partition2, state[walker])
                        DGRANDE = KineticAB(N, M, ch, sh, partition2,
                                       partition1, state[walker]) / Kq_new
                    # Perhaps this one should not be optimized
                    Ratio_two[walker] += np.min(np.array([1, DGRANDE]))
                    Ratio[walker] += DGRANDE
                    Ratio_two_all[walker, n_used[walker]]\
                                        += np.min(np.array([1, DGRANDE]))
                    Ratio_all[walker, n_used[walker]] += DGRANDE

                    n_used[walker] += 1

    print(exp / NUMBER_OF_STEPS) # all of them should be accepted!
    print('av size of cluster: ',
                        AVSIZ / NUMBER_OF_STEPS / N_WALKERS)
    if TYPE_OF_OBS == 0:
        return state, T0_all, T1_all, T0_avg / n_used, T1_avg / n_used
    if TYPE_OF_OBS == 1:
        return state, Ratio_two_all, Ratio_all,\
                              Ratio_two / n_used, Ratio / n_used


def MAIN_integral(betaJ, zeta, N, r, NUMBER_OF_STEPS, start,
                    SAMPLING_NUMBER, CODE, RENYI_ORDER, partition2,
                            partition1, N_WALKERS, cle0, cle1, qq, index):
   '''This is an example of how one should run the code for
   the thermodynamic integration. In our calculations this function
   is called many times in parallel and hence results and parameters
   are collected and saved in qq'''

   InitialState, IS = np.zeros((5, 5, 5), dtype=np.bool_), 0
   rezultat = MontePath_Master(betaJ, zeta, N, r, InitialState, IS,
                      NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, partition2, partition1, N_WALKERS,
                      cle0, cle1, TYPE_OF_OBS_STR = 'integral')
   
   np.save(SAVETO + 'T0all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_cle0_' + str(cle0)\
            + '_cle1_' + str(cle1) + '_integral_Part_' + str(partition2)\
            + '_' + str(partition1), rezultat[-4])
   np.save(SAVETO + 'T1all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_cle0_' + str(cle0)\
            + '_cle1_' + str(cle1) + '_integral_Part_' + str(partition2)\
            + '_' + str(partition1), rezultat[-3])
   

   qq.put((index, cle0, cle1, rezultat[-2], rezultat[-1]))
   return rezultat[-2], rezultat[-1]


def MAIN_ratio(betaJ, zeta, N, r, NUMBER_OF_STEPS, start,
                    SAMPLING_NUMBER, CODE, RENYI_ORDER, partition2,
                                          partition1, N_WALKERS, qq, index):
   '''This is an example of how one should run the code for
   the evaluatoin of SWAP. In our calculations this function
   is called many times in parallel and hence results and parameters
   are collected and saved in qq. Notice that MontePath_Master() is
   called two times!'''

   InitialState, IS = np.zeros((5, 5, 5), dtype=np.bool_), 0
   rezultat1 = MontePath_Master(betaJ, zeta, N, r, InitialState, IS,
                      NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, partition2, partition1, N_WALKERS,
                      0, 1, TYPE_OF_OBS_STR = 'ratio')
   rezultat2 = MontePath_Master(betaJ, zeta, N, r, InitialState, IS,
                      NUMBER_OF_STEPS, start, SAMPLING_NUMBER, CODE,
                      RENYI_ORDER, partition2, partition1, N_WALKERS,
                      1, 0, TYPE_OF_OBS_STR = 'ratio')

   np.save(SAVETO + 'BOL0all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_ratio_Part_'\
            + str(partition2) + '_' + str(partition1), rezultat1[-4])
   np.save(SAVETO + 'BOL1all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_ratio_Part_'\
            + str(partition2) + '_' + str(partition1), rezultat2[-4])
   np.save(SAVETO + 'MET0all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_ratio_Part_'\
            + str(partition2) + '_' + str(partition1), rezultat1[-3])
   np.save(SAVETO + 'MET1all_betaJ_' + str(betaJ) + 'zeta_' + str(zeta)\
            + 'r_' + str(r) + '_N_' + str(N) + '_ratio_Part_'\
            + str(partition2) + '_' + str(partition1), rezultat2[-3])


   qq.put((partition1, rezultat1[-2], rezultat2[-2],
                                        rezultat1[-1], rezultat2[-1]))
   return rezultat1[-1], rezultat2[-1]



# This is a very fast run and in here to compile numba before
# the run. 
betaJ = 1
zeta = betaJ / 5 # number of beads is 5
N = 4
r = 0.1
NUMBER_OF_STEPS = 10 ** 3
start = 10 ** 2
SAMPLING_NUMBER = 10
CODE = 0
N_WALKERS = 1
RENYI_ORDER = 2
partition1 = 0
partition2 = 2
cle0, cle1 = 1, 0
TYPE_OF_OBS = 'integral'
InitialState, IS = np.zeros((5, 5, 5), dtype=np.bool_), 0
ruru = MontePath_Master(betaJ, zeta, N, r, InitialState, IS, 
                      NUMBER_OF_STEPS, start, SAMPLING_NUMBER,
                      CODE, RENYI_ORDER, partition2, partition1,
                      N_WALKERS, cle0, cle1, TYPE_OF_OBS)



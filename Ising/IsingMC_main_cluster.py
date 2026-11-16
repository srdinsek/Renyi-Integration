# -*- coding: utf-8 -*-
"""
This is Wolf cluster algorithm adapted for the computation of
entanglement Renyi entropy in 1D Quantum Ising Model. It is
primarelly meant for the calculation of the second order
Renyi entropy, but it can be easily adapted for higher orders.

Main function cluster_monte_carlo, can be used for any (connected)
partition size and thermodynamic integration. Thus also for
SWAP operator.

For the two observables investigated in the paper, we use
rutines:
1) IsingMC_RAT.py  (for evaluation by SWAP)
2) IsingMC_INT.py  (for evaluation by thermodynamic integration)
each of them running the same algorithm, but for different set of
parameters and observables.

This code was written for the pourpose of comparing the integration
method to the SWAP method at large n_spins.

The main algorithm strongly depends on subrutines defined in
IsingMC_main_lib. It's performance is enhaced by the extensive
use of numba's decorator @njit.
"""

import os
import numpy as np
from numba import njit
from numpy.random import randint,seed

from IsingMC_main_lib import expand_cluster,observable,kinetic_ab


@njit(error_model="numpy")
def cluster_monte_carlo(betaJ, zeta, n_spins, r, initial_state, initial_seed,
                        n_steps, start, sampling_interval, sampling_offset,
                        renyi_order, partition2, partition1, n_walkers,
                        lam0, lam1, observable_type):
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
    multiwalkers algorithm. Number of walkers is set by n_walkers.
    '''
    # betaJ = inverse temperature
    # zeta = betaJ / n_beads
    # n_spins = number of particles
    # r = strenght of the magnetic field
    # initial_state = initial state (used only if initial_seed != 0)
    # initial_seed = seed, but random state is used only if initial_seed = 0
    # n_steps = number of clusters produced
    # start = number of steps when sampling starts
    # sampling_interval = frequency of sampling
    # sampling_offset = sampling phase shift. Usually set to 0
    # renyi_order = order of Renyi entropy. Only 2 can be used
    # n_walkers = number of independent walkers.
    # lam0 = weight on the springs that fulfil the split copies
    # boundary condition
    # lam1 = weight on the sping joining two copies.
    # observable_type = either 'integral' or 'ratio' - specifies
    # the observables that we want to sample - energy or SWAP.

    if observable_type == 'integral':

        observable_id = 0

    if observable_type == 'ratio':

        observable_id = 1

    # Because clusters are biger at small r, we
    # make less moves. Hence n_steps given as argument is multiplied
    # by 60 for r > 3 and by 10 for 1 < r <= 3, and the sampling
    # interval is set to 10 for r < 1.
    if r > 3:
        n_steps = n_steps * 60
    elif r > 1:
        n_steps = n_steps * 10
    elif r < 1:
        sampling_interval = 10

    # zeta is redefined, so that n_beads is indeed an integer
    n_beads = int(betaJ / zeta)
    print('M : ', n_beads)
    zeta = betaJ / n_beads

    # Constants used in cluster algorithm
    ch, sh = np.cosh(betaJ * r / n_beads), np.sinh(betaJ * r / n_beads)
    time_coupling = np.log(1 / np.tanh(betaJ * r / n_beads)) / 2
    # link in imaginary-time dir. - "kinetic"
    p_time = 1 - np.exp(-2 * time_coupling)
    # link along the bead - "potential"
    p_space = 1 - np.exp(-2 * betaJ / n_beads)
    # link1 in imag.-time at the boundary
    p_boundary1 = 1 - np.exp(-2 * time_coupling * lam0)
    # link2 in imag.-time at the boundary
    p_boundary2 = 1 - np.exp(-2 * time_coupling * lam1)
    # only renyi_order = 2 is supported
    n_slices = renyi_order * n_beads
    n_sites = renyi_order * n_beads * n_spins
    print(ch, sh, time_coupling)

    # initial state for each walker
    if initial_seed > 0:
        seed(initial_seed)
        state = initial_state.astype(np.bool_)
    else:
        seed(0)
        state = randint(0, 2, (n_walkers, n_spins,
                               n_slices)).astype(np.bool_)

    # counts number of sample points for each walker
    n_used = np.zeros(n_walkers, dtype=np.int64)

    # averages of observables
    # number of accepted moves
    n_accepted = np.zeros((n_walkers), dtype=np.int64)
    T0_avg = np.zeros((n_walkers), dtype=np.float64)
    T1_avg = np.zeros((n_walkers), dtype=np.float64)
    swap_avg = np.zeros((n_walkers), dtype=np.float64) #SWAP
    swap_min_avg = np.zeros((n_walkers), dtype=np.float64)

    # full observables
    T0_samples = np.zeros((n_walkers,
          (n_steps - start) // sampling_interval), dtype=np.float64)
    T1_samples = np.zeros((n_walkers,
          (n_steps - start) // sampling_interval), dtype=np.float64)
    swap_samples = np.zeros((n_walkers,
          (n_steps - start) // sampling_interval), dtype=np.float64)
    swap_min_samples = np.zeros((n_walkers,
          (n_steps - start) // sampling_interval), dtype=np.float64)

    # some arrays that help reducing memorry used
    cluster_sites = np.zeros((n_walkers, n_sites, 2), dtype=np.int64)
    cluster_ids = np.zeros((n_walkers, n_sites), dtype=np.int64)
    pocket = np.zeros((n_walkers, n_sites), dtype=np.int64)

    # Parameters used for the cluster algortihm
    total_cluster_size = 0

    # We start the Metropolis loop
    for step in range(n_steps):
        # Each walker is treated separately
        for walker in range(n_walkers):

            ''' We construct the new move. '''
            #-------------------------------------------
            # Wolf cluster move
            #-------------------------------------------

            # randomly choosen starting spin at position (n, m)
            n = randint(0, n_spins)
            m = randint(0, n_slices)

            # empty the array
            cluster_sites.fill(0)

            # add the spin (n, m) to the cluster
            cluster_sites[walker, 0, 0] = n
            cluster_sites[walker, 0, 1] = m

            # add the spin (n, m) to the pocket in number format
            pocket[walker, 0] = m + n * n_slices

            # save the position in a number format
            cluster_ids[walker, 0] = m + n * n_slices

            # size of the pocket
            pocket_size = 1
            # size of the cluster
            cluster_size = 1

            # loop over the pocket until it is empty
            while pocket_size > 0:
                # read the position of the last spin in the pocket
                n = pocket[walker, pocket_size - 1] // n_slices
                m = pocket[walker, pocket_size - 1] % n_slices

                # Check the neighbors of the spin from the pocket
                # and add them to the cluster with the physical
                # probabilities with the function expand_cluster()
                cluster_size, pocket[walker], pocket_size,\
                cluster_sites[walker], cluster_ids[walker] = expand_cluster(
                              n_spins, n_beads, n_slices, time_coupling,
                              p_space, p_time, p_boundary1, p_boundary2,
                              partition1, partition2, state[walker],
                              pocket[walker], cluster_sites[walker],
                              pocket_size, cluster_size, n, m,
                              cluster_ids[walker])
                # The used spin is removed from the pocket, but
                # the new ones are added. They are added also
                # to cluster_sites

            # Flip the cluster
            for i in range(cluster_size):
                n, m = cluster_sites[walker, i, 0], cluster_sites[walker, i, 1]
                state[walker, n, m] = not state[walker, n, m]
            # every step is accepted
            n_accepted[walker] += 1
            total_cluster_size += cluster_size

            ''' Here we meassure our observable. '''
            if step % sampling_interval == sampling_offset and step > start:
                if observable_id == 0:

                    # we record the energy of the spings along the path
                    T0, T1 = observable(n_spins, n_beads, time_coupling,
                                    partition1, partition2, state[walker])
                    T0_avg[walker] += T0
                    T1_avg[walker] += T1
                    T0_samples[walker, n_used[walker]] += T0
                    T1_samples[walker, n_used[walker]] += T1

                    n_used[walker] += 1

                if observable_id == 1:

                    # we evaluate the SWAP operator
                    if lam0 == 0:
                        K_ab = kinetic_ab(n_spins, n_beads, ch, sh,
                                          partition1, partition2,
                                          state[walker])
                        swap_ratio = kinetic_ab(n_spins, n_beads, ch, sh,
                                            partition2, partition1,
                                                state[walker]) / K_ab
                        swap_ratio = 1 / swap_ratio
                    else:
                        K_ab = kinetic_ab(n_spins, n_beads, ch, sh,
                                          partition1, partition2,
                                          state[walker])
                        swap_ratio = kinetic_ab(n_spins, n_beads, ch, sh,
                                       partition2, partition1,
                                       state[walker]) / K_ab
                    # Perhaps this one should not be optimized
                    swap_min_avg[walker] += np.min(np.array([1, swap_ratio]))
                    swap_avg[walker] += swap_ratio
                    swap_min_samples[walker, n_used[walker]]\
                                        += np.min(np.array([1, swap_ratio]))
                    swap_samples[walker, n_used[walker]] += swap_ratio

                    n_used[walker] += 1

    print(n_accepted / n_steps) # all of them should be accepted!
    print('av size of cluster: ',
                        total_cluster_size / n_steps / n_walkers)
    if observable_id == 0:
        return state, T0_samples, T1_samples, T0_avg / n_used,\
                              T1_avg / n_used
    if observable_id == 1:
        return state, swap_min_samples, swap_samples,\
                              swap_min_avg / n_used, swap_avg / n_used


def run_integral(betaJ, zeta, n_spins, r, n_steps, start,
                    sampling_interval, sampling_offset, renyi_order,
                    partition2, partition1, n_walkers, lam0, lam1, queue,
                    index, saveto):
   '''This is an example of how one should run the code for
   the thermodynamic integration. In our calculations this function
   is called many times in parallel and hence results and parameters
   are collected and saved in queue'''

   initial_state, initial_seed = np.zeros((5, 5, 5), dtype=np.bool_), 0
   result = cluster_monte_carlo(betaJ, zeta, n_spins, r, initial_state,
                      initial_seed, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, partition2, partition1,
                      n_walkers, lam0, lam1, observable_type = 'integral')

   tag = ('betaJ_' + str(betaJ) + 'zeta_' + str(zeta) + 'r_' + str(r)
          + '_N_' + str(n_spins) + '_cle0_' + str(lam0) + '_cle1_'
          + str(lam1) + '_integral_Part_' + str(partition2) + '_'
          + str(partition1))
   np.save(os.path.join(saveto, 'T0all_' + tag), result[-4])
   np.save(os.path.join(saveto, 'T1all_' + tag), result[-3])

   queue.put((index, lam0, lam1, result[-2], result[-1]))
   return result[-2], result[-1]


def run_ratio(betaJ, zeta, n_spins, r, n_steps, start,
                    sampling_interval, sampling_offset, renyi_order,
                    partition2, partition1, n_walkers, queue, index, saveto):
   '''This is an example of how one should run the code for
   the evaluatoin of SWAP. In our calculations this function
   is called many times in parallel and hence results and parameters
   are collected and saved in queue. Notice that cluster_monte_carlo() is
   called two times!'''

   initial_state, initial_seed = np.zeros((5, 5, 5), dtype=np.bool_), 0
   result1 = cluster_monte_carlo(betaJ, zeta, n_spins, r, initial_state,
                      initial_seed, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, partition2, partition1,
                      n_walkers, 0, 1, observable_type = 'ratio')
   result2 = cluster_monte_carlo(betaJ, zeta, n_spins, r, initial_state,
                      initial_seed, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, partition2, partition1,
                      n_walkers, 1, 0, observable_type = 'ratio')

   tag = ('betaJ_' + str(betaJ) + 'zeta_' + str(zeta) + 'r_' + str(r)
          + '_N_' + str(n_spins) + '_ratio_Part_' + str(partition2) + '_'
          + str(partition1))
   for prefix, samples in [('BOL0all_', result1[-4]),
                           ('BOL1all_', result2[-4]),
                           ('MET0all_', result1[-3]),
                           ('MET1all_', result2[-3])]:
      np.save(os.path.join(saveto, prefix + tag), samples)

   queue.put((partition1, result1[-2], result2[-2],
                                        result1[-1], result2[-1]))
   return result1[-1], result2[-1]



# This is a very fast run and in here to compile numba before
# the run.
betaJ = 1
zeta = betaJ / 5 # number of beads is 5
n_spins = 4
r = 0.1
n_steps = 10 ** 3
start = 10 ** 2
sampling_interval = 10
sampling_offset = 0
n_walkers = 1
renyi_order = 2
partition1 = 0
partition2 = 2
lam0, lam1 = 1, 0
observable_type = 'integral'
initial_state, initial_seed = np.zeros((5, 5, 5), dtype=np.bool_), 0
warmup = cluster_monte_carlo(betaJ, zeta, n_spins, r, initial_state,
                      initial_seed, n_steps, start, sampling_interval,
                      sampling_offset, renyi_order, partition2, partition1,
                      n_walkers, lam0, lam1, observable_type)

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:12:04 2021

@author: Miha

In this program function run_renyi() is defined. It can be used to
evaluate Rényi entropy of second order for any system of distinguishable
particles in analytical potential. Although the ergodicity of the sampling
is not guaranteed. It is primarelly meant for the calculation of the
second order Renyi entropy, but it can be easily adapted for higher orders.

For a realistic 2d potential, linear interpolation is implemented.

It's performance is enhaced by the extensive use of numba's decorator
@njit, but at the same time numba restricts us with it limitations.

It is important to define:
saveto - where to save the results (argument of run_renyi)
POTLOC - if realistic potential is used, it's address
DISLOC - if realistic potential is used, the address of it's grid

"""

import numpy as np
import numba
from numba import jit, njit
from numpy.random import rand,randint,normal,uniform,seed,permutation
import os
import math
import pickle
from scipy.interpolate import bisplrep, bisplev

HERE = os.path.dirname(os.path.abspath(__file__))
POTLOC = os.path.join(HERE, "potentials.npy")
DISLOC = os.path.join(HERE, "distances.npy")

@njit(error_model="numpy")
def initial_energy(path, b, b2, v1, v2, n_beads, partition1, partition2,
                   mass, mu, lamda, potential_kind, B, potential, coor, dk,
                   lam1, lam2, c1, c2, potential_correction):
    '''This function evaluates full energy at the begining of the
    MC simulation. '''

    kinetic_part1, kinetic_part2, V = 0, 0, 0
    for i in range(len(path[0])):
        b.fill(0)
        b2.fill(0)
        v1.fill(0)
        v2.fill(0)
        b += path[:, i]
        b2 += path[:, (i-1) % n_beads + n_beads * (i // n_beads)]
        b2[:partition1] = path[:partition1, i - 1]
        v1 += (b - b2)
        b2[partition1 :partition2] = path[partition1 :partition2, i - 1]
        v2 += (b - b2)
        if i % n_beads == 0:
           # this if statement was added in 26.4.2021
           kinetic_part1 += (v1 @ mass @ (v1 * lam1))
           kinetic_part1 += (v2 @ mass @ (v2 * lam2))
        else:
           # we can use only one, since they must be equal
           # alternatively one could multiply one wiht lambda
           # and another one with (1 - lambda)
           kinetic_part1 += (v2 @ mass @ v2)

        # This terms is used ony if potential_correction == True
        if i % n_beads == 0 and potential_correction == True:
           V += potential_energy(b, mu, lamda, potential_kind, B,
                                 potential, coor, dk) / 2

           b[:partition1] = path[:partition1, i - n_beads]
           V += potential_energy(b, mu, lamda, potential_kind, B,
                           potential, coor, dk) / 2 * c1

           b[partition1 :partition2] = path[partition1 :partition2,
                                            i - n_beads]
           V += potential_energy(b, mu, lamda, potential_kind, B,
                           potential, coor, dk) / 2 * c2
        else:
           V += potential_energy(b, mu, lamda, potential_kind, B,
                                 potential, coor, dk)
    K1 = kinetic_part1
    K2 = kinetic_part2
    E01 = K1 + V
    E02 = K2 + V
    return V, K1, K2, E01, E02

@njit(error_model="numpy")
def potential_energy(vector, mu, lamda, potential_kind, B, potential, coor,
                     dk):
    '''Potential energy of the configuration. This function can be
    changed baased on the needs.'''

    if potential_kind == 0:
        # Analytical potential. This one is written for x^4
        square = vector ** 2
        general = vector @ mu @ vector / 2 + (square @ lamda @ square)
        return general

    if potential_kind == 1:
        # Linear interpolation, for 2D potential. It is written
        # like this, because there are no rutines in numbe for
        # splines.
        i, j = 0, 0
        x, y = vector[0], vector[1]

        # This is for extrapolation
        if x > coor[-1]:
            i = len(coor) - 1
        if x <= coor[0]:
            i = 1
        if y > coor[-1]:
            j = len(coor) - 1
        if y <= coor[0]:
            j = 1

        # find nearest i, j to actual position
        if j == 0 and i == 0:
            indices = np.searchsorted(coor, vector)
            i, j = indices[0], indices[1]
        elif j == 0:
            indices = np.searchsorted(coor, y)
            j = indices
        elif i == 0:
            indices = np.searchsorted(coor, x)
            i = indices

        x1, x2 = coor[i - 1], coor[i]
        y1, y2 = coor[j - 1], coor[j]

        # linear interpolation and extrapolation outside of the edges.
        general = potential[i - 1, j - 1] * (x2 - x) * (y2 - y)\
        + potential[i, j - 1] * (x - x1) * (y2 - y)\
        + potential[i - 1, j] * (x2 - x) * (y - y1)\
        + potential[i, j] * (x - x1) * (y - y1)
        return general / dk

@njit(error_model="numpy")
def multi(mat, vec, weight):
   '''Just multiplication with weight on one side. '''
   #this thing works only if mat is diagonal. If mat is not diagonal
   # then we should multiply both sides!
   return vec @ mat @ (vec * weight)

@njit(error_model="numpy")
def sample_replicas(beta, zeta, lam1, lam2, renyi_order, mass, mu, lamda,
                    shift, partition1, partition2, n_steps, epsilon,
                    start, random_seed, sampling_offset, sampling_interval,
                    state, potential, coor, dk, dimension=1, memory=False,
                    position=False, random_state=True, potential_kind=1):
    '''This is the function that simulates the ensembles that gives
    the Rényi entanglement entropy. Parameter ´partition1´ specifies
    the number of joint copies starting from particle 0. Zero means there
    are none. Parameter ´partition2´ specifies the trial size starting
    from particle 0. If ´partition1´ is nonzero, then the trial size is
    partition2 - partition1. Trial copies are the ones that we
    modify during the thermodynamic integration. For example function
    with parameters partition2, partition1=0 simulates split ensemble
    and modifies the subsystem spaning from particle 0 to particle
    (partition2 - 1). The function with parameters partition2=0, partition1
    simulates ensemlbe, where copies spanning from particle 0 to particle
    (partition1 - 1) are joined. Parameter ´potential_kind´ selects the
    potential: 0 for analytic, 1 for realistic 2D (linear interpolation).'''

    # zeta - beta / n_beads
    # epsilon - define the variance of random numbers
    # mu, lamda and mass are matrices, that set the interaction
    # between the particles.
    # The interaction in the imaginary-time direciton is default, while the
    # potential energy can be made arbitrary by changing the function
    # potential_energy. In imaginary-time direction only the nearest
    # neighbours can interact.
    # The copies don't appear as separate subarrays, but are simply padded, so
    # that effective number of beads is n_beads * renyi_order.
    # memory and position declare if we want to recorde history
    # of measurements.

    #-------------------------------------------
    # Constants and some arrays
    #-------------------------------------------

    potential_correction = False
    B = 4 * 40 * 0.1 ** 6
    epsilon_class = 1 / epsilon
    n_beads = max(int(beta / zeta), 3)
    path_length = n_beads * renyi_order
    print('M : ', n_beads)
    zeta = beta / n_beads
    c1 = lam1
    lam1 = np.ones(dimension, dtype=np.float64) / 2
    lam1[:partition2] = lam1[:partition2] * c1 * 2
    lam1[:partition1] = np.zeros(partition1, dtype=np.float64)
    c2 = lam2
    lam2 = np.ones(dimension, dtype=np.float64) / 2
    lam2[:partition2] = lam2[:partition2] * c2 * 2
    lam2[:partition1] = np.ones(partition1, dtype=np.float64)

    reflection_mask = np.zeros(dimension, dtype=np.float64)
    reflection_mask[:partition2] = np.ones(partition2, dtype=np.float64) * c2
    reflection_mask[:partition1] = np.ones(partition1, dtype=np.float64)

    lam_mask = np.zeros(dimension, dtype=np.float64)
    lam_mask[partition1 : partition2] = np.ones(partition2 - partition1,
                                                            dtype=np.float64)
    identity = np.ones(dimension, dtype=np.float64)
    seed(random_seed)
    mu, lamda, mass = mu * zeta, lamda * zeta, mass / zeta
    # mu, lamad and mass are matrices
    if n_beads <= 2:
        print('Change zeta, M is smaller or equal to 2!')
    else:
        # Initial state
        if random_state:
            path = normal(0, 1, (dimension, n_beads * renyi_order))
            for i in range(dimension):
                path[i] += shift[i]
        else:
            path = np.copy(state)

        # Some arrays used during the calcualtion
        path_new = np.zeros((dimension, n_beads * renyi_order),
                            dtype=np.float64)
        b = np.zeros(dimension, dtype=np.float64)
        b2 = np.zeros(dimension, dtype=np.float64)
        v1 = np.zeros(dimension, dtype=np.float64)
        v2 = np.zeros(dimension, dtype=np.float64)

        #-------------------------------------------
        # Initial kinetic and potential energy
        #-------------------------------------------
        V, K1, K2, E01, E02 = initial_energy(path, b, b2, v1, v2, n_beads,
                                         partition1, partition2, mass,
                                         mu, lamda, potential_kind, B,
                                         potential, coor, dk, lam1, lam2,
                                         c1, c2, potential_correction)

        #-------------------------------------------
        # Arrays for storing the observables
        #-------------------------------------------

        if memory:
            V_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            K_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            ratio_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            T0_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            T1_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            V0_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
            V1_samples = np.zeros((n_steps - start) // sampling_interval,
                                                             dtype=np.float32)
        if position:
            x = np.zeros(((n_steps - start) // sampling_interval,
                          dimension, n_beads * renyi_order),
                         dtype=np.float32)

        n_accepted, n_used = 0, 0
        average_V, average_K1, average_ratio = 0, 0, 0
        new_bead = np.zeros((dimension, renyi_order), dtype=np.float64)
        old_bead = np.zeros((dimension, renyi_order), dtype=np.float64)
        next_bead = np.zeros((dimension, renyi_order), dtype=np.float64)
        previous_bead = np.zeros((dimension, renyi_order), dtype=np.float64)
        displacement = np.zeros((dimension, renyi_order), dtype=np.float64)
        pot_q = np.zeros(dimension, dtype=np.float64)
        pot_d = np.zeros(dimension, dtype=np.float64)

        #-------------------------------------------
        # Metropolos loop (from here on numba is usefull)
        #-------------------------------------------
        for step in range(1, n_steps + 1):

            # Just slightly perturb one random bead
            if step % 100 != 0:
                displacement.fill(0)

                # create perturbation for each copy
                j = randint(0, n_beads, renyi_order)
                while (j[0] == 0 and j[1] == n_beads - 1)\
                        or (j[0] == n_beads - 1 and j[1] == 0)\
                                or (j[0] == 0 and j[1] == 0):
                    j = randint(0, n_beads, renyi_order)

                # the perturbing vectors
                displacement += uniform(-epsilon, epsilon,
                                        (dimension, renyi_order))

                # prepare quantities for the calculation
                # and calculate the change in "energy"
                dV, dK1 = 0, 0
                # loop over the copies
                for al in range(renyi_order):
                    jjj = j[al]
                    jj = jjj + n_beads * al

                    # perturbed bead
                    new_bead[:, al] = path[:, jj] + displacement[:, al]
                    # unperturbed one
                    old_bead[:, al] = path[:, jj]
                    # neighbouring beads
                    next_bead[:, al] = path[:, (jjj + 1) % n_beads
                                            + n_beads * al]
                    previous_bead[:, al] = path[:, (jjj - 1) % n_beads
                                                + n_beads * al]

                    #-------------------------------------------
                    # Change in the potential energy
                    #-------------------------------------------
                    # Very simple if potential_correction == False
                    if jj % n_beads == 0 and potential_correction == True:
                       pot_q.fill(0)
                       pot_d.fill(0)
                       dV += (potential_energy(new_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor, dk) \
                            - potential_energy(old_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor,
                                    dk)) / 2
                       pot_q += new_bead[:, al]
                       pot_d += old_bead[:, al]
                       pot_q[:partition1] = path[:partition1, jj - n_beads]
                       pot_d[:partition1] = path[:partition1, jj - n_beads]
                       dV += (potential_energy(new_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor, dk) \
                            - potential_energy(old_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor,
                                    dk)) / 2 * c1
                       pot_q[partition1 :partition2] = path[
                                        partition1 :partition2, jj - n_beads]
                       pot_d[partition1 :partition2] = path[
                                        partition1 :partition2, jj - n_beads]
                       dV += (potential_energy(new_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor, dk) \
                            - potential_energy(old_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor,
                                    dk)) / 2 * c2
                    else:
                       dV += potential_energy(new_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor, dk) \
                            - potential_energy(old_bead[:, al], mu, lamda,
                                    potential_kind, B, potential, coor, dk)

                    #-------------------------------------------
                    # Change in the kinetic energy
                    #-------------------------------------------
                    next_bead[:partition1, al] = path[:partition1,
                                                      (jj + 1) % path_length]
                    previous_bead[:partition1, al] = path[:partition1,
                                                      (jj - 1) % path_length]

                    #====================FIRST===========================
                    if jj % n_beads == 0:
                       # this if statement was added in 26.4.2021
                       dK1 += (multi(mass, previous_bead[:, al]
                                     - new_bead[:, al], lam1)\
                            - multi(mass, previous_bead[:, al]
                                    - old_bead[:, al], lam1))\
                            + multi(mass, next_bead[:, al]
                                    - new_bead[:, al], identity)\
                            - multi(mass, next_bead[:, al]
                                    - old_bead[:, al], identity)
                    elif jj % n_beads == (n_beads - 1):
                       dK1 += multi(mass, previous_bead[:, al]
                                    - new_bead[:, al], identity)\
                            - multi(mass, previous_bead[:, al]
                                    - old_bead[:, al], identity)\
                            + (multi(mass, next_bead[:, al]
                                     - new_bead[:, al], lam1)\
                            - multi(mass, next_bead[:, al]
                                    - old_bead[:, al], lam1))
                    else:
                       dK1 += 2 * ((new_bead[:, al] - old_bead[:, al]) @ mass \
                            @ (new_bead[:, al] + old_bead[:, al]
                               - next_bead[:, al] - previous_bead[:, al]))
                    #====================================================

                    next_bead[partition1 :partition2, al] = path[
                                    partition1 :partition2,
                                    (jj + 1) % path_length]
                    previous_bead[partition1 :partition2, al] = path[
                                    partition1 :partition2,
                                    (jj - 1) % path_length]

                    #====================SECOND==========================
                    if jj % n_beads == 0:
                       # this if statement was added in 26.4.2021
                       dK1 += (multi(mass, previous_bead[:, al]
                                     - new_bead[:, al], lam2)\
                             - multi(mass, previous_bead[:, al]
                                     - old_bead[:, al], lam2))
                    elif jj % n_beads == (n_beads - 1):
                       dK1 += (multi(mass, next_bead[:, al]
                                     - new_bead[:, al], lam2)\
                             - multi(mass, next_bead[:, al]
                                     - old_bead[:, al], lam2))
                    #====================================================

                dE1 = dV + dK1

                # Accept the move if
                if dE1 < 0:
                    for al in range(renyi_order):
                        path[:, j[al] + al * n_beads] = new_bead[:, al]
                    n_accepted += 1

                    V += dV
                    K1 += dK1

                elif rand() < np.exp(-dE1):
                    for al in range(renyi_order):
                        path[:, j[al] + al * n_beads] = new_bead[:, al]
                    n_accepted += 1

                    V += dV
                    K1 += dK1

            # the mirroring of the replica
            else:
                # this was added in 30.6.2021

                dV, dK1 = 0, 0
                old_bead[:, 0] = path[:, n_beads]
                new_bead[:, 0] = -path[:, n_beads]
                previous_bead[:, 0] = path[:, n_beads - 1]

                old_bead[:, 1] = path[:, -1]
                new_bead[:, 1] = -path[:, -1]
                previous_bead[:, 1] = path[:, 0]

                # loop over copies
                for al in range(2):
                    #====================ONLY_===========================
                    dK1 += (multi(mass, previous_bead[:, al]\
                                    - new_bead[:, al], reflection_mask)\
                          - multi(mass, previous_bead[:, al]\
                                    - old_bead[:, al], reflection_mask))
                    #====================================================

                dE1 = dV + dK1

                # Accept the move if
                if dE1 < 0:
                    path[:, n_beads:] = -path[:, n_beads:]
                    n_accepted += 1

                    V += dV
                    K1 += dK1

                elif rand() < np.exp(-dE1):
                    path[:, n_beads:] = -path[:, n_beads:]
                    n_accepted += 1

                    V += dV
                    K1 += dK1

            #-------------------------------------------
            # Observables
            #-------------------------------------------
            ''' Here we meassure our observable. '''
            if step > start and step % sampling_interval == sampling_offset:
                average_V += V
                average_K1 -= K1

                kinetic_part1, kinetic_part2 = 0, 0
                potential_part1, potential_part2 = 0, 0
                # this works only in renyi_order = 2 example!!
                for al in range(renyi_order):
                    b.fill(0)
                    b2.fill(0)
                    v1.fill(0)
                    v2.fill(0)

                    b += path[:, al * n_beads]
                    b2 += path[:, (al + 1) * n_beads - 1]
                    b2[:partition1] = path[:partition1, al * n_beads - 1]
                    v1 += (b - b2) * lam_mask
                    kinetic_part1 += (v1 @ mass @ v1) # lam1

                    b2[partition1 :partition2] = path[\
                                    partition1 :partition2, al * n_beads - 1]
                    v2 += (b - b2) * lam_mask
                    kinetic_part2 += (v2 @ mass @ v2) # lam2

                    b[:partition1] = path[:partition1, al * n_beads - n_beads]
                    potential_part1 += potential_energy(b, mu, lamda,
                                    potential_kind, B, potential, coor,
                                    dk) / 2
                    b[partition1 :partition2] = path[\
                              partition1 :partition2, al * n_beads - n_beads]
                    potential_part2 += potential_energy(b, mu, lamda,
                                    potential_kind, B, potential, coor,
                                    dk) / 2

                kinetic_difference = (kinetic_part2 - kinetic_part1)
                average_ratio += kinetic_difference

                if memory:
                    V_samples[n_used] = V
                    K_samples[n_used] = K1
                    ratio_samples[n_used] = kinetic_difference
                    T0_samples[n_used] = kinetic_part1
                    T1_samples[n_used] = kinetic_part2
                    V0_samples[n_used] = potential_part1
                    V1_samples[n_used] = potential_part2
                if position:
                    x[n_used] = np.copy(path)
                n_used += 1

    if not memory:
        K_samples = np.zeros(10, dtype=np.float32)
        V_samples = np.zeros(10, dtype=np.float32)
        ratio_samples = np.zeros(10, dtype=np.float32)
        T0_samples = np.zeros(10, dtype=np.float32)
        T1_samples = np.zeros(10, dtype=np.float32)
        V0_samples = np.zeros(10, dtype=np.float32)
        V1_samples = np.zeros(10, dtype=np.float32)
    if not position:
        x = np.zeros((10, 10, 10), dtype=np.float32)
    print('acceptance ratio: ', n_accepted / n_steps)
    return path, average_V / (n_used*beta), average_K1 / (n_used*beta), x,\
                    K_samples / beta, V_samples / beta, T0_samples,\
                    T1_samples, average_ratio / n_used, ratio_samples,\
                    V0_samples, V1_samples


def run_renyi(beta, zeta, lam1, renyi_order, mass, mu, lamda, shift,
              partition1, partition2, n_steps, epsilon, start, random_seed,
              sampling_offset, sampling_interval, dimension=1, memory=False,
              position=False, index=False, name='lam6_doublew', lam2=False,
              real_index=10, potential_type='analytic', *, saveto):

    '''This function calls sample_replicas() and saves the
    results. In the case of realistic 2D potential it performs some
    spline interpolation, to make grid for linear interpolation denser.
    saveto is the folder where the results are saved.'''

    if partition1 == False:
        partition = partition2

    # these must be converted to numpy arrays
    mass = np.array(mass, dtype=np.float64)
    mu = np.array(mu, dtype=np.float64)
    lamda = np.array(lamda, dtype=np.float64)
    shift = np.array(shift, dtype=np.float64)
    state = np.zeros((2, 2), dtype=np.float64)

    #-------------------------------------------
    # Call the main function
    #-------------------------------------------

    if potential_type == 'realistic':
        # This is done like this, because the spline interpolation is not
        # yet implemented in numba. We thought that this is easier to do,
        # than implementing spline only for this pourpose.

        # Constants have to be evaluated like in the main function
        Mw = max(int(beta / zeta), 3)
        zetaw = beta / Mw
        epsilonw = np.sqrt(zetaw)

        Mw = max(int(beta / zetaw), 3)
        print('M : ', Mw)
        zetaw = beta / Mw
        muw, lamdaw, massw = mu * zetaw, lamda * zetaw, mass / zetaw

        # we load the realistic potential
        potentials = np.load(POTLOC)
        distances = np.load(DISLOC)
        # These parameters, depend on the potential in use.
        n_grid = 30
        x, dx = np.linspace(0.5, distances[real_index][0] - 0.5,
                                                    n_grid, retstep=True)
        x = x - (x[-1] - x[0]) / 2 - x[0]
        X, Y = np.meshgrid(x, x)
        # fit the spline to the potential
        if real_index == 24:
           spline = bisplrep(X, Y, potentials[real_index] * zetaw, s=0.00001)
        else:
           spline = bisplrep(X, Y, potentials[real_index] * zetaw, s=0)

        # interpolate the potential with the spline, to make points denser
        n_grid = 10000
        coor, dk = np.linspace(0.5, distances[real_index][0] - 0.5,
                                                    n_grid, retstep=True)
        coor = coor - (coor[-1] - coor[0]) / 2 - coor[0]
        dk = dk ** 2
        potential = bisplev(coor, coor, tck=spline)

        results = sample_replicas(beta, zeta, lam1, lam2, renyi_order, mass,
                   mu, lamda, shift, partition1, partition2, n_steps,
                   epsilon, start, random_seed, sampling_offset,
                   sampling_interval, state, potential, coor, dk,
                   dimension=dimension, memory=memory, position=position,
                   random_state=True, potential_kind=1)

    elif potential_type == 'analytic':

        # just define the constants that are not used, but simplify the
        # program, so that now only one potential function can be used.
        # Alternative would be some kind of decorator, but it could be
        # that numba would complicate things.
        dk = 0
        coor = np.array((2, 2), dtype=np.float64)
        potential = np.zeros((2, 2), dtype=np.float64)
        results = sample_replicas(beta, zeta, lam1, lam2, renyi_order, mass,
                   mu, lamda, shift, partition1, partition2, n_steps,
                   epsilon, start, random_seed, sampling_offset,
                   sampling_interval, state, potential, coor, dk,
                   dimension=dimension, memory=memory, position=position,
                   random_state=True, potential_kind=0)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    (path, average_V, average_K1, x, K_samples, V_samples, T0_samples,
     T1_samples, average_ratio, ratio_samples, V0_samples,
     V1_samples) = results
    prefix = os.path.join(saveto, name + 'INDEX_' + str(index))
    for suffix, data in [('_Ratio', average_ratio),
                         ('_Ratio_tot', ratio_samples),
                         ('_T0_tot', T0_samples), ('_T1_tot', T1_samples),
                         ('_K_tot', K_samples), ('_V_tot', V_samples),
                         ('_path', path), ('_x', x),
                         ('_V0_tot', V0_samples), ('_V1_tot', V1_samples)]:
        np.save(prefix + suffix, data)

    #-------------------------------------------
    # Saving the parameters
    #-------------------------------------------

    dictionary = {'beta': beta,
              'zeta': zeta,
              'clé': lam1,
              'cle2': lam2,
              'order': renyi_order,
              'mass': mass,
              'mu': mu,
              'lamda': lamda,
              'shift': shift,
              'partition1': partition1,
              'partition2': partition2,
              'NUMBER_OF_STEPS': n_steps,
              'epsilon': epsilon,
              'start': start,
              'memory': memory,
              'position': position,
              'SEED': random_seed,
              'CODE': sampling_offset,
              'SAMPLING_NUMBER': sampling_interval,
              'dimension': dimension
              }

    with open(os.path.join(saveto, name + '_args.p'), 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=4)

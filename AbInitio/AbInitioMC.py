# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:12:04 2021

@author: Miha

In this program function CHQO_ND_Renyi() is defined. It can be used to
evaluate Rényi entropy of second order for any system of distinguishable
particles in analytical potential. Although the ergodicity of the sampling
is not guaranteed. It is primarelly meant for the calculation of the
second order Renyi entropy, but it can be easily adapted for higher orders.

For a realistic 2d potential, linear interpolation is implemented.

It's performance is enhaced by the extensive use of numba's decorator
@njit, but at the same time numba restricts us with it limitations.

It is important to define:
SAVETO - where to save the results
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

SAVETO = '/scratchalpha/srdinsek/HarmonicOscillator/CHQO_ND_Renyi'
POTLOC = "potentials.npy"
DISLOC = "distances.npy"

@njit(error_model="numpy")
def InitialEnergy(baza, b, b2, v1, v2, M, partition1, partition2,
                            mass, mu, lamda, A, B, potential, coor, dk,
                                                cle1, cle2, c1, c2, POT_CORR):
    '''This function evaluates full energy at the begining of the
    MC simulation. '''
                                 
    kinetic_part1, kinetic_part2, V = 0, 0, 0
    for i in range(len(baza[0])):
        b.fill(0)
        b2.fill(0)
        v1.fill(0)
        v2.fill(0)
        b += baza[:, i]
        b2 += baza[:, (i-1) % M + M * (i // M)]
        b2[:partition1] = baza[:partition1, i - 1]
        v1 += (b - b2)
        b2[partition1 :partition2] = baza[partition1 :partition2, i - 1]
        v2 += (b - b2)
        if i % M == 0:
           # this if statement was added in 26.4.2021
           kinetic_part1 += (v1 @ mass @ (v1 * cle1))
           kinetic_part1 += (v2 @ mass @ (v2 * cle2))
        else:
           # kinetic_part1 += (v1 @ mass @ v1)
           # we can use only one, since they must be equal
           # alternatively one could multiply one wiht lambda
           # and another one with (1 - lambda)
           kinetic_part1 += (v2 @ mass @ v2)
        
        # This terms is used ony if POT_CORR == True
        if i % M == 0 and POT_CORR == True:
           V += V_standard(b, mu, lamda, A, B, potential, coor, dk) / 2

           b[:partition1] = baza[:partition1, i - M]
           V += V_standard(b, mu, lamda, A, B,
                           potential, coor, dk) / 2 * c1

           b[partition1 :partition2] = baza[partition1 :partition2, i - M]
           V += V_standard(b, mu, lamda, A, B,
                           potential, coor, dk) / 2 * c2
        else:
           V += V_standard(b, mu, lamda, A, B, potential, coor, dk)
    K1 = kinetic_part1 # / 2
    K2 = kinetic_part2
    E01 = K1 + V
    E02 = K2 + V
    return V, K1, K2, E01, E02

@njit(error_model="numpy")
def V_standard(vector, mu, lamda, A, B, potential, coor, dk):
    '''Potential energy of the configuration. This function can be
    changed baased on the needs.'''

    if A == 0:
        # Analytical potential. This one is written for x^4
        square = vector ** 2
        general = vector @ mu @ vector / 2 + (square @ lamda @ square)
        return general

    if A == 1:
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
def multi(mat, vec, cle):
   '''Just multiplication with weight cle on one side. '''
   #this thing works only if mat is diagonal. If mat is not diagonal
   # then we should multiply both sides!
   return vec @ mat @ (vec * cle)

@njit(error_model="numpy")
def functionND_copies_dc(beta, zeta, cle1, cle2, order, mass, mu, lamda,
                      shift, partition1, partition2, NUMBER_OF_STEPS, epsilon,
                      start, SEED, CODE, SAMPLING_NUMBER, state, potential,
                      coor, dk, dimension=1, memory=False, position=False,
                      RandomState=True):
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
    (partition1 - 1) are joined.'''
    
    # zeta - beta / M
    # epsilon - define the variance of random numbers
    # mu, lamda and mass are matrices, that set the interaction
    # between the particles.
    # The interaction in the imaginary-time direciton is default, while the
    # potential energy can be made arbitrary by changing the function
    # V_standard. In imaginary-time direction only the nearest neighbours can
    # interact.
    # The copies don't appear as separate subarrays, but are simply padded, so
    # that effective $M$ is $M * alpha$, where $alpha$ is the order of
    # Renyi entropy.
    # memory and position declare if we want to recorde history
    # of measurements.

    #-------------------------------------------
    # Constants and some arrays
    #-------------------------------------------

    POT_CORR = False
    A, B = 1, 4 * 40 * 0.1 ** 6
    epsilon_class = 1 / epsilon
    M = max(int(beta / zeta), 3)
    LENGHT = M * order
    print('M : ', M)
    zeta = beta / M
    c1 = cle1
    cle1 = np.ones(dimension, dtype=np.float64) / 2
    cle1[:partition2] = cle1[:partition2] * c1 * 2
    cle1[:partition1] = np.zeros(partition1, dtype=np.float64)
    c2 = cle2
    cle2 = np.ones(dimension, dtype=np.float64) / 2
    cle2[:partition2] = cle2[:partition2] * c2 * 2
    cle2[:partition1] = np.ones(partition1, dtype=np.float64)

    reflection_mask = np.zeros(dimension, dtype=np.float64)
    reflection_mask[:partition2] = np.ones(partition2, dtype=np.float64) * c2
    reflection_mask[:partition1] = np.ones(partition1, dtype=np.float64)

    cle_mask = np.zeros(dimension, dtype=np.float64)
    cle_mask[partition1 : partition2] = np.ones(partition2 - partition1,
                                                            dtype=np.float64)
    identiteta = np.ones(dimension, dtype=np.float64)
    seed(SEED)
    mu, lamda, mass = mu * zeta, lamda * zeta, mass / zeta
    # mu, lamad and mass are matrices
    if M <= 2:
        print('Change zeta, M is smaller or equal to 2!')
    else:
        # Initial state
        if RandomState:
            baza = normal(0, 1, (dimension, M * order))
            for i in range(dimension):
                baza[i] += shift[i]
        else:
            baza = np.copy(state)

        # Some arrays used during the calcualtion
        baza_q = np.zeros((dimension, M * order), dtype=np.float64)
        b = np.zeros(dimension, dtype=np.float64)
        b2 = np.zeros(dimension, dtype=np.float64)
        v1 = np.zeros(dimension, dtype=np.float64)
        v2 = np.zeros(dimension, dtype=np.float64)

        #-------------------------------------------
        # Initial kinetic and potential energy
        #-------------------------------------------
        V, K1, K2, E01, E02 = InitialEnergy(baza, b, b2, v1, v2, M,
                                         partition1, partition2, mass,
                                         mu, lamda, A, B, potential,
                                         coor, dk, cle1, cle2, c1, c2,
                                         POT_CORR)

        #-------------------------------------------
        # Arrays for storing the observables
        #-------------------------------------------

        if memory:
            V_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            K_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            Ratio_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            T0_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            T1_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            V0_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            V1_tot = np.zeros((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                                             dtype=np.float32)
            #E[0] = E0
        if position:
            x = np.zeros(((NUMBER_OF_STEPS - start) // SAMPLING_NUMBER,
                                      dimension, M * order), dtype=np.float32)

        n_excepted, n_used = 0, 0
        average_V, average_K1, average_ratio = 0, 0, 0
        q = np.zeros((dimension, order), dtype=np.float64)
        d = np.zeros((dimension, order), dtype=np.float64)
        dp1 = np.zeros((dimension, order), dtype=np.float64)
        dm1 = np.zeros((dimension, order), dtype=np.float64)
        motnja = np.zeros((dimension, order), dtype=np.float64)
        pot_q = np.zeros(dimension, dtype=np.float64)
        pot_d = np.zeros(dimension, dtype=np.float64)

        #-------------------------------------------
        # Metropolos loop (from here on numba is usefull)
        #-------------------------------------------
        for step in range(1, NUMBER_OF_STEPS + 1):

            # Just slightly perturb one random bead
            if step % 100 != 0:
                motnja.fill(0)

                # create perturbation for each copy
                j = randint(0, M, order)
                while (j[0] == 0 and j[1] == M - 1)\
                        or (j[0] == M - 1 and j[1] == 0)\
                                or (j[0] == 0 and j[1] == 0):
                    j = randint(0, M, order)

                # the perturbing vectors
                motnja += uniform(-epsilon, epsilon, (dimension, order))# * q

                # prepare quantities for the calculation
                # and calculate the change in "energy"
                dV, dK1 = 0, 0
                # loop over the copies
                for al in range(order):
                    jjj = j[al]
                    jj = jjj + M * al
                    
                    # perturbed bead
                    q[:, al] = baza[:, jj] + motnja[:, al]
                    # unperturbed one
                    d[:, al] = baza[:, jj]
                    # neighbouring beads
                    dp1[:, al] = baza[:, (jjj + 1) % M + M * al]
                    dm1[:, al] = baza[:, (jjj - 1) % M + M * al]

                    #-------------------------------------------
                    # Change in the potential energy
                    #-------------------------------------------
                    # Very simple if POT_CORR == False
                    if jj % M == 0 and POT_CORR == True:
                       pot_q.fill(0)
                       pot_d.fill(0)
                       dV += (V_standard(q[:, al], mu, lamda, A, B,
                                            potential, coor, dk) \
                            - V_standard(d[:, al], mu, lamda, A, B,
                                            potential, coor, dk)) / 2
                       pot_q += q[:, al]
                       pot_d += d[:, al]
                       pot_q[:partition1] = baza[:partition1, jj - M]
                       pot_d[:partition1] = baza[:partition1, jj - M]
                       dV += (V_standard(q[:, al], mu, lamda, A, B,
                                            potential, coor, dk) \
                            - V_standard(d[:, al], mu, lamda, A, B,
                                            potential, coor, dk)) / 2 * c1
                       pot_q[partition1 :partition2] = baza[
                                                partition1 :partition2, jj - M]
                       pot_d[partition1 :partition2] = baza[
                                                partition1 :partition2, jj - M]
                       dV += (V_standard(q[:, al], mu, lamda, A, B,
                                            potential, coor, dk) \
                            - V_standard(d[:, al], mu, lamda, A, B,
                                            potential, coor, dk)) / 2 * c2
                    else:
                       dV += V_standard(q[:, al], mu, lamda, A, B,
                                            potential, coor, dk) \
                            - V_standard(d[:, al], mu, lamda, A, B,
                                            potential, coor, dk)

                    #-------------------------------------------
                    # Change in the kinetic energy
                    #-------------------------------------------
                    dp1[:partition1, al] = baza[:partition1, (jj + 1) % LENGHT]
                    dm1[:partition1, al] = baza[:partition1, (jj - 1) % LENGHT]

                    #====================FIRST===========================
                    if jj % M == 0:
                       # this if statement was added in 26.4.2021
                       dK1 += (multi(mass, dm1[:, al] - q[:, al], cle1)\
                            - multi(mass, dm1[:, al] - d[:, al], cle1))\
                            + multi(mass, dp1[:, al] - q[:, al], identiteta)\
                            - multi(mass, dp1[:, al] - d[:, al], identiteta)
                    elif jj % M == (M - 1):
                       dK1 += multi(mass, dm1[:, al] - q[:, al], identiteta)\
                            - multi(mass, dm1[:, al] - d[:, al], identiteta)\
                            + (multi(mass, dp1[:, al] - q[:, al], cle1)\
                            - multi(mass, dp1[:, al] - d[:, al], cle1))
                    else:
                       dK1 += 2 * ((q[:, al] - d[:, al]) @ mass \
                            @ (q[:, al] + d[:, al] - dp1[:, al] - dm1[:, al]))
                    #====================================================

                    dp1[partition1 :partition2, al] = baza[
                                    partition1 :partition2, (jj + 1) % LENGHT]
                    dm1[partition1 :partition2, al] = baza[
                                    partition1 :partition2, (jj - 1) % LENGHT]

                    #====================SECOND==========================
                    if jj % M == 0:
                       # this if statement was added in 26.4.2021
                       dK1 += (multi(mass, dm1[:, al] - q[:, al], cle2)\
                             - multi(mass, dm1[:, al] - d[:, al], cle2))
                    elif jj % M == (M - 1):
                       dK1 += (multi(mass, dp1[:, al] - q[:, al], cle2)\
                             - multi(mass, dp1[:, al] - d[:, al], cle2))
                    #====================================================

                dE1 = dV + dK1

                # Accept the move if
                if dE1 < 0:
                    for al in range(order):
                        baza[:, j[al] + al * M] = q[:, al]
                    n_excepted += 1

                    V += dV
                    K1 += dK1

                elif rand() < np.exp(-dE1):
                    for al in range(order):
                        baza[:, j[al] + al * M] = q[:, al]
                    n_excepted += 1

                    V += dV
                    K1 += dK1

            # the mirroring of the replica
            else:
                # this was added in 30.6.2021

                dV, dK1 = 0, 0
                d[:, 0] = baza[:, M]
                q[:, 0] = -baza[:, M]
                dm1[:, 0] = baza[:, M - 1]

                d[:, 1] = baza[:, -1]
                q[:, 1] = -baza[:, -1]
                dm1[:, 1] = baza[:, 0]

                # loop over copies
                for al in range(2):
                    #====================ONLY_===========================
                    dK1 += (multi(mass, dm1[:, al]\
                                    - q[:, al], reflection_mask)\
                          - multi(mass, dm1[:, al]\
                                    - d[:, al], reflection_mask))
                    #====================================================

                dE1 = dV + dK1

                # Accept the move if
                if dE1 < 0:
                    baza[:, M:] = -baza[:, M:]
                    n_excepted += 1

                    V += dV
                    K1 += dK1

                elif rand() < np.exp(-dE1):
                    baza[:, M:] = -baza[:, M:]
                    n_excepted += 1

                    V += dV
                    K1 += dK1

            #-------------------------------------------
            # Observables
            #-------------------------------------------
            ''' Here we meassure our observable. '''
            if step > start and step % SAMPLING_NUMBER == CODE:
                average_V += V
                average_K1 -= K1

                kinetic_part1, kinetic_part2 = 0, 0
                potential_part1, potential_part2 = 0, 0
                # for link in range(M):
                # this works only in order = 2 example!!
                for al in range(order):
                    b.fill(0)
                    b2.fill(0)
                    v1.fill(0)
                    v2.fill(0)

                    b += baza[:, al * M]
                    b2 += baza[:, (al + 1) * M - 1]
                    b2[:partition1] = baza[:partition1, al * M - 1]
                    v1 += (b - b2) * cle_mask
                    kinetic_part1 += (v1 @ mass @ v1) # cle1

                    b2[partition1 :partition2] = baza[\
                                        partition1 :partition2, al * M - 1]
                    v2 += (b - b2) * cle_mask
                    kinetic_part2 += (v2 @ mass @ v2) # cle2

                    b[:partition1] = baza[:partition1, al * M - M]
                    potential_part1 += V_standard(b, mu, lamda, A, B,
                                                potential, coor, dk) / 2
                    b[partition1 :partition2] = baza[\
                                        partition1 :partition2, al * M - M]
                    potential_part2 += V_standard(b, mu, lamda, A, B,
                                                potential, coor, dk) / 2

                DGRANDE = (kinetic_part2 - kinetic_part1)
                average_ratio += DGRANDE

                if memory:
                    V_tot[n_used] = V
                    K_tot[n_used] = K1
                    Ratio_tot[n_used] = DGRANDE
                    T0_tot[n_used] = kinetic_part1
                    T1_tot[n_used] = kinetic_part2
                    V0_tot[n_used] = potential_part1
                    V1_tot[n_used] = potential_part2
                if position:
                    x[n_used] = np.copy(baza)
                n_used += 1

    if not memory:
        K_tot = np.zeros(10, dtype=np.float32)
        V_tot = np.zeros(10, dtype=np.float32)
        Ratio_tot = np.zeros(10, dtype=np.float32)
        T0_tot = np.zeros(10, dtype=np.float32)
        T1_tot = np.zeros(10, dtype=np.float32)
        V0_tot = np.zeros(10, dtype=np.float32)
        V1_tot = np.zeros(10, dtype=np.float32)
    if not position:
        x = np.zeros((10, 10, 10), dtype=np.float32)
    print('exceptance ratio: ', n_excepted / NUMBER_OF_STEPS)
    return baza, average_V / (n_used*beta), average_K1 / (n_used*beta), x,\
                            K_tot / beta, V_tot / beta, T0_tot, T1_tot,\
                            average_ratio / n_used, Ratio_tot, V0_tot, V1_tot


def CHQO_ND_Renyi(beta, zeta, cle, order, mass, mu, lamda, shift, partition1,
                  partition2, NUMBER_OF_STEPS, epsilon, start, SEED, CODE,
                  SAMPLING_NUMBER, dimension=1, memory=False, position=False,
                  index=False, NAME='lam6_doublew', cle2=False,
                  real_index=10, POTENTIAL_TYPE='analytic'):

    '''This function calls functionND_copies_dc() and saves the
    results. In the case of realistic 2D potential it performs some
    spline interpolation, to make grid for linear interpolation denser.'''

    if partition1 == False:
        parition = partition2

    # these must be converted to numpy arrays
    mass = np.array(mass, dtype=np.float64)
    mu = np.array(mu, dtype=np.float64)
    lamda = np.array(lamda, dtype=np.float64)
    shift = np.array(shift, dtype=np.float64)
    state = np.zeros((2, 2), dtype=np.float64)

    #-------------------------------------------
    # Call the main function
    #-------------------------------------------
    
    if POTENTIAL_TYPE == 'realistic':
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
        N = 30
        x, dx = np.linspace(0.5, distances[real_index][0] - 0.5,
                                                            N, retstep=True)
        x = x - (x[-1] - x[0]) / 2 - x[0]
        X, Y = np.meshgrid(x, x)
        # fit the spline to the potential
        if real_index == 24:
           spline = bisplrep(X, Y, potentials[real_index] * zetaw, s=0.00001)
        else:
           spline = bisplrep(X, Y, potentials[real_index] * zetaw, s=0)

        # interpolate the potential with the spline, to make points denser
        N = 10000
        coor, dk = np.linspace(0.5, distances[real_index][0] - 0.5,
                                                            N, retstep=True)
        coor = coor - (coor[-1] - coor[0]) / 2 - coor[0]
        dk = dk ** 2
        potential = bisplev(coor, coor, tck=spline)

        vse = functionND_copies_dc(beta, zeta, cle, cle2, order, mass, mu,
                   lamda, shift, partition1, partition2, NUMBER_OF_STEPS,
                   epsilon, start, SEED, CODE,
                   SAMPLING_NUMBER, state, potential, coor, dk,
                   dimension=dimension, memory=memory, position=position,
                   RandomState=True)

    elif POTENTIAL_TYPE == 'analytic':

        # just define the constants that are not used, but simplify the
        # program, so that now only one potential function can be used.
        # Alternative would be some kind of decorator, but it could be
        # that numba would complicate things.
        dk = 0
        coor = np.array((2, 2), dtype=np.float64)
        vse = functionND_copies_dc(beta, zeta, cle, cle2, order, mass, mu,
                   lamda, shift, partition1, partition2, NUMBER_OF_STEPS,
                   epsilon, start, SEED, CODE,
                   SAMPLING_NUMBER, state, potential, coor, dk,
                   dimension=dimension, memory=memory, position=position,
                   RandomState=True)

    #-------------------------------------------
    # Saving the results
    #-------------------------------------------

    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_Ratio'),
    vse[-4])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_Ratio_tot'),
    vse[-3])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_T0_tot'),
    vse[-6])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_T1_tot'),
    vse[-5])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_K_tot'),
    vse[-8])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_V_tot'),
    vse[-7])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_basis'),
    vse[0])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_x'),
    vse[-9])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_V0_tot'),
    vse[-2])
    np.save(os.path.join(SAVETO, NAME + 'INDEX_' + str(index) + '_V1_tot'),
    vse[-1])

    #-------------------------------------------
    # Saving the parameters
    #-------------------------------------------

    dictionary = {'beta': beta,
              'zeta': zeta,
              'clé': cle,
              'cle2': cle2,
              'order': order,
              'mass': mass,
              'mu': mu,
              'lamda': lamda,
              'shift': shift,
              'partition1': partition1,
              'partition2': partition2,
              'NUMBER_OF_STEPS': NUMBER_OF_STEPS,
              'epsilon': epsilon,
              'start': start,
              'memory': memory,
              'position': position,
              'SEED': SEED,
              'CODE': CODE,
              'SAMPLING_NUMBER': SAMPLING_NUMBER,
              'dimension': dimension
              }

    with open(os.path.join(SAVETO, NAME + '_args.p'), 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=4)
    # with open(os.path.join(SAVETO, NAME + '_args.p'), 'rb') as fp:
    #     dictionary = pickle.load(fp)




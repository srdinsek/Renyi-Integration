'''
This is a library of functions that are used by the Wolff
cluster algorithm and heat-bath algorithm.
'''

import math
import numba
import numpy as np
from numba import jit, njit
from numpy.random import rand


@njit(fastmath=True)
def isin(val, arr, SIZE):
    '''Alternative to "is in".'''
    for i in range(SIZE):
        if arr[i] == val:
            return True
    return False

#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# Observables
#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

@njit
def observable(N, M, const, partition1, partition2, state):
    '''Observable in thermodynamic integration.'''
    # the contributing ones are just (0, M-1), (0, 2M - 1), 
    # (M, 2M - 1), (M, M-1)
    T0, T1 = 0, 0
    # we sample only this section
    for x in range(partition1, partition2):
        if state[x, 0] == state[x, M - 1]:
            T0 += 1
        else:
            T0 -= 1
        if state[x, M] == state[x, 2 * M - 1]:
            T0 += 1
        else:
            T0 -= 1
        if state[x, 0] == state[x, 2 * M - 1]:
            T1 += 1
        else:
            T1 -= 1
        if state[x, M] == state[x, M - 1]:
            T1 += 1
        else:
            T1 -= 1
    return T0 * const, T1 * const

@njit
def KineticAB(N, M, ch, sh, partition2, partition1, state):
    '''Observable for SWAP operator. '''
    K = 1
    if partition1 > partition2:
       start = partition2
       end = partition1
    else:
       start = partition1
       end = partition2
    for i in range(start, end):
        if i < partition2:
            for j in [-1, M - 1]:
                if state[i, j] == state[i, (j + 1) % (2 * M)]:
                    K *= ch
                else:
                    K *= sh
        else:
            j = M - 1
            if state[i, j] == state[i, (j + 1) % M]:
                K *= ch
            else:
                K *= sh
            if state[i, j + M] == state[i, (j + 1) % M + M]:
                K *= ch
            else:
                K *= sh
    return K

#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# Wolf Cluster
#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

@njit
def OPERATOR(nn, mm, SIZE_C, POCKET, SIZE, CLUSTER_indi, NUMO, CLUSTER_name):
    '''This function adds new spin to the cluster. '''
    #CLUSTER[nn, mm] = True
    CLUSTER_indi[SIZE_C, 0] = nn
    CLUSTER_indi[SIZE_C, 1] = mm
    CLUSTER_name[SIZE_C] = NUMO
    SIZE_C += 1
    POCKET[SIZE] = NUMO
    SIZE += 1
    return SIZE_C, POCKET, SIZE, CLUSTER_indi, CLUSTER_name

@njit
def EXPAND(N, M, TIME, const, pP, pK, pK1, pK2, partition1, partition2,
              state, POCKET, CLUSTER_indi, SIZE, SIZE_C, n, m, CLUSTER_name):
    '''The main part of the Wolf cluster algorithm. Adding the spin in the
    vertical and horisontal dimension is split. Because the neighbouring
    spins in the same bead interact only by the potential term and are
    not affected by the boundary condition.  '''
    NUMBER = POCKET[SIZE - 1]
    POCKET[SIZE - 1] = 0
    SIZE -= 1

    #-------------------------------------------
    # Neighbouring spins in the same bead
    #-------------------------------------------

    nn = (n - 1) % N
    mm = m
    NUMO = mm + nn * TIME
    if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pP:
        SIZE_C, POCKET, SIZE,\
        CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                              SIZE, CLUSTER_indi, NUMO,
                                                            CLUSTER_name)

    nn = (n + 1) % N
    mm = m
    NUMO = mm + nn * TIME
    if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pP:
        SIZE_C, POCKET, SIZE,\
        CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                              SIZE, CLUSTER_indi, NUMO,
                                                            CLUSTER_name)

    #---------------------------------------------------
    # Neighbouring spins in the imaginary-time direction
    #---------------------------------------------------

    if n < partition1:
        #-------------------------------------------
        # the connected replicas coordinates
        #-------------------------------------------
        nn = n
        mm = (m + 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm = (m - 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

    elif n < partition2 and m % M == 0:
        #-------------------------------------------
        # mixed coordinates on one side
        #-------------------------------------------
        nn = n
        mm = (m + 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm = (m - 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK2:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm = (m - 1) % M + (m // M ) * M
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK1:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

    elif n < partition2 and m % M == (M - 1):
        #-------------------------------------------
        # mixed coordinates on the other side
        #-------------------------------------------
        nn = n
        mm = (m - 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm = (m + 1) % TIME
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK2:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm = (m + 1) % M + (m // M ) * M
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK1:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

    else:
        #-------------------------------------------
        # split replicas coordinates
        #-------------------------------------------
        nn = n
        mm = (m + 1) % M + (m // M ) * M
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                 and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

        nn = nn
        mm =  (m - 1) % M + (m // M ) * M
        NUMO = mm + nn * TIME
        if state[nn, mm] == state[n, m]\
                 and (not isin(NUMO, CLUSTER_name, SIZE_C)) and rand() < pK:
            SIZE_C, POCKET, SIZE,\
            CLUSTER_indi, CLUSTER_name = OPERATOR(nn, mm, SIZE_C, POCKET,
                                                  SIZE, CLUSTER_indi, NUMO,
                                                               CLUSTER_name)

    return SIZE_C, POCKET, SIZE, CLUSTER_indi, CLUSTER_name


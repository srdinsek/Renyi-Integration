'''
This is a library of functions that are used by the Wolff
cluster algorithm and heat-bath algorithm.
'''

import numpy as np
from numba import njit
from numpy.random import rand


@njit(fastmath=True)
def isin(val, arr, size):
    '''Alternative to "is in": checks the first size elements of arr.'''
    for i in range(size):
        if arr[i] == val:
            return True
    return False

#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# Observables
#-------------------------------------------
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

@njit
def observable(n_spins, n_beads, time_coupling, partition1, partition2,
               state):
    '''Observable in thermodynamic integration.'''
    # the contributing ones are just (0, n_beads - 1), (0, 2 n_beads - 1),
    # (n_beads, 2 n_beads - 1), (n_beads, n_beads - 1)
    T0, T1 = 0, 0
    # we sample only this section
    for x in range(partition1, partition2):
        if state[x, 0] == state[x, n_beads - 1]:
            T0 += 1
        else:
            T0 -= 1
        if state[x, n_beads] == state[x, 2 * n_beads - 1]:
            T0 += 1
        else:
            T0 -= 1
        if state[x, 0] == state[x, 2 * n_beads - 1]:
            T1 += 1
        else:
            T1 -= 1
        if state[x, n_beads] == state[x, n_beads - 1]:
            T1 += 1
        else:
            T1 -= 1
    return T0 * time_coupling, T1 * time_coupling

@njit
def kinetic_ab(n_spins, n_beads, ch, sh, partition2, partition1, state):
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
            for j in [-1, n_beads - 1]:
                if state[i, j] == state[i, (j + 1) % (2 * n_beads)]:
                    K *= ch
                else:
                    K *= sh
        else:
            j = n_beads - 1
            if state[i, j] == state[i, (j + 1) % n_beads]:
                K *= ch
            else:
                K *= sh
            if (state[i, j + n_beads]
                    == state[i, (j + 1) % n_beads + n_beads]):
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
def add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                   cluster_sites, site_id, cluster_ids):
    '''This function adds new spin to the cluster. '''
    cluster_sites[cluster_size, 0] = n_new
    cluster_sites[cluster_size, 1] = m_new
    cluster_ids[cluster_size] = site_id
    cluster_size += 1
    pocket[pocket_size] = site_id
    pocket_size += 1
    return cluster_size, pocket, pocket_size, cluster_sites, cluster_ids

@njit
def expand_cluster(n_spins, n_beads, n_slices, time_coupling, p_space, p_time,
                   p_boundary1, p_boundary2, partition1, partition2, state,
                   pocket, cluster_sites, pocket_size, cluster_size, n, m,
                   cluster_ids):
    '''The main part of the Wolf cluster algorithm. Adding the spin in the
    vertical and horisontal dimension is split. Because the neighbouring
    spins in the same bead interact only by the potential term and are
    not affected by the boundary condition.  '''
    pocket[pocket_size - 1] = 0
    pocket_size -= 1

    #-------------------------------------------
    # Neighbouring spins in the same bead
    #-------------------------------------------

    n_new = (n - 1) % n_spins
    m_new = m
    site_id = m_new + n_new * n_slices
    if (state[n_new, m_new] == state[n, m]
            and not isin(site_id, cluster_ids, cluster_size)
            and rand() < p_space):
        cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
            add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                           cluster_sites, site_id, cluster_ids)

    n_new = (n + 1) % n_spins
    m_new = m
    site_id = m_new + n_new * n_slices
    if (state[n_new, m_new] == state[n, m]
            and not isin(site_id, cluster_ids, cluster_size)
            and rand() < p_space):
        cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
            add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                           cluster_sites, site_id, cluster_ids)

    #---------------------------------------------------
    # Neighbouring spins in the imaginary-time direction
    #---------------------------------------------------

    if n < partition1:
        #-------------------------------------------
        # the connected replicas coordinates
        #-------------------------------------------
        n_new = n
        m_new = (m + 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new = (m - 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

    elif n < partition2 and m % n_beads == 0:
        #-------------------------------------------
        # mixed coordinates on one side
        #-------------------------------------------
        n_new = n
        m_new = (m + 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new = (m - 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_boundary2):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new = (m - 1) % n_beads + (m // n_beads ) * n_beads
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_boundary1):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

    elif n < partition2 and m % n_beads == (n_beads - 1):
        #-------------------------------------------
        # mixed coordinates on the other side
        #-------------------------------------------
        n_new = n
        m_new = (m - 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new = (m + 1) % n_slices
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_boundary2):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new = (m + 1) % n_beads + (m // n_beads ) * n_beads
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_boundary1):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

    else:
        #-------------------------------------------
        # split replicas coordinates
        #-------------------------------------------
        n_new = n
        m_new = (m + 1) % n_beads + (m // n_beads ) * n_beads
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

        n_new = n_new
        m_new =  (m - 1) % n_beads + (m // n_beads ) * n_beads
        site_id = m_new + n_new * n_slices
        if (state[n_new, m_new] == state[n, m]
                and not isin(site_id, cluster_ids, cluster_size)
                and rand() < p_time):
            cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
                add_to_cluster(n_new, m_new, cluster_size, pocket, pocket_size,
                               cluster_sites, site_id, cluster_ids)

    return cluster_size, pocket, pocket_size, cluster_sites, cluster_ids


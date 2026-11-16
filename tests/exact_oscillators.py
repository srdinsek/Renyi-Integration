"""
Exact second order Renyi entropy of two coupled harmonic oscillators.

The discretised path integral sampled by AbInitioMC has the action
x^T A x, where x contains the positions of all beads of the two replicas.
Because the action is quadratic, Z = const / sqrt(det A), and the Renyi
entropy for the same number of beads is

    S2 = -ln(Z_joined / Z_split) = (ln det A_joined - ln det A_split) / 2.

In the continuum limit the entropy of the full system is given by
Eq. (C9) of M. Srdinsek et al., Phys. Rev. B (2023),
"Renyi entropy of a quantum anharmonic chain at nonzero temperature".
"""

import numpy as np

N_REPLICAS = 2


def action_matrix(beta, n_beads, mass, mu, partition2, w_split, w_joined):
    '''Matrix A of the action x^T A x, for the potential x mu x / 2. For
    particles < partition2 the spring that closes the ring of each replica
    has weight w_split and the spring that joins the two replicas has
    weight w_joined. Bead k of particle p is element p * size + k.'''
    zeta = beta / n_beads
    dimension, size = len(mu), N_REPLICAS * n_beads
    A = np.zeros((dimension * size, dimension * size))

    def spring(p, k, l, weight):
        a, b = p * size + k, p * size + l
        c = weight * mass[p][p] / zeta
        A[a, a] += c
        A[b, b] += c
        A[a, b] -= c
        A[b, a] -= c

    for p in range(dimension):
        for k in range(size):
            replica, j = divmod(k, n_beads)
            own = replica * n_beads + n_beads - 1 if j == 0 else k - 1
            if j == 0 and p < partition2:
                spring(p, k, own, w_split)
                spring(p, k, (k - 1) % size, w_joined)
            else:
                spring(p, k, own, 1)
    for k in range(size):
        for p in range(dimension):
            for q in range(dimension):
                A[p * size + k, q * size + k] += zeta * mu[p][q] / 2
    return A


def log_z(beta, n_beads, mass, mu, partition2, lam):
    '''ln Z along the path with weights (1 - lam)^3 and lam^3, up to a
    constant that does not depend on lam.'''
    A = action_matrix(beta, n_beads, mass, mu, partition2,
                      (1 - lam) ** 3, lam ** 3)
    return -np.linalg.slogdet(A)[1] / 2


def exact_discrete_s2(beta, n_beads, mass, mu, partition2):
    '''S2 of particles < partition2 for n_beads beads.'''
    return log_z(beta, n_beads, mass, mu, partition2, 0)\
           - log_z(beta, n_beads, mass, mu, partition2, 1)


def exact_continuum_full_s2(beta, m, mu):
    '''S2 of the full system in the continuum limit, Eq. (C9) of the PRB
    paper, with hbar = 1 and normal mode frequencies of H = p^2 / 2m +
    x mu x / 2.'''
    frequencies = np.sqrt(np.linalg.eigvalsh(np.array(mu) / m))
    return -np.sum(np.log(np.tanh(frequencies * beta / 2)))

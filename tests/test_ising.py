"""
Tests of the functions used inside the Ising cluster algorithm.

Run from the repository root with: python -m unittest discover tests
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Ising'))
from IsingMC_main_lib import isin, observable, kinetic_ab
from IsingMC_main_lib import add_to_cluster, expand_cluster
import IsingMC_INT
import IsingMC_RAT

N_SPINS = 4
N_BEADS = 5
RENYI_ORDER = 2
N_SLICES = RENYI_ORDER * N_BEADS
N_SITES = N_SLICES * N_SPINS
TIME_COUPLING = 0.7
CH, SH = 1.3, 0.4
ALWAYS, NEVER = 1.0, 0.0
LINKS_PER_SPIN = 2  # two boundary links per spin and observable
BULK_SPIN, BULK_BEAD = 1, 2  # far from the boundaries in imaginary time
NEW_SPIN, NEW_BEAD = 2, 3  # spin added to the cluster in test_add_to_cluster
INT_ARGUMENTS = 10  # arguments of IsingMC_INT.py without saveto
RAT_ARGUMENTS = 9  # arguments of IsingMC_RAT.py without saveto


def replicas_opposite():
    '''First replica all up, second replica all down.'''
    state = np.zeros((N_SPINS, N_SLICES), dtype=np.bool_)
    state[:, :N_BEADS] = True
    return state


class TestObservables(unittest.TestCase):

    def test_isin(self):
        arr = np.array([3, 7, 9, 11])
        self.assertTrue(isin(9, arr, 3))
        self.assertFalse(isin(11, arr, 3))  # only the first 3 are checked

    def test_observable_aligned(self):
        state = np.ones((N_SPINS, N_SLICES), dtype=np.bool_)
        partition1, partition2 = 0, 2
        T0, T1 = observable(N_SPINS, N_BEADS, TIME_COUPLING, partition1,
                            partition2, state)
        expected = LINKS_PER_SPIN * (partition2 - partition1) * TIME_COUPLING
        self.assertAlmostEqual(T0, expected)
        self.assertAlmostEqual(T1, expected)

    def test_observable_opposite_replicas(self):
        # links inside each replica agree, links between replicas do not
        partition1, partition2 = 0, 2
        T0, T1 = observable(N_SPINS, N_BEADS, TIME_COUPLING, partition1,
                            partition2, replicas_opposite())
        expected = LINKS_PER_SPIN * (partition2 - partition1) * TIME_COUPLING
        self.assertAlmostEqual(T0, expected)
        self.assertAlmostEqual(T1, -expected)

    def test_kinetic_ab(self):
        partition2, partition1 = 2, 0
        n_links = LINKS_PER_SPIN * (partition2 - partition1)
        aligned = np.ones((N_SPINS, N_SLICES), dtype=np.bool_)
        self.assertAlmostEqual(kinetic_ab(N_SPINS, N_BEADS, CH, SH,
                               partition2, partition1, aligned),
                               CH ** n_links)
        self.assertAlmostEqual(kinetic_ab(N_SPINS, N_BEADS, CH, SH,
                               partition2, partition1, replicas_opposite()),
                               SH ** n_links)


class TestCluster(unittest.TestCase):

    def start_cluster(self, n, m):
        '''Cluster and pocket that contain only the spin (n, m).'''
        site = m + n * N_SLICES
        pocket = np.zeros(N_SITES, dtype=np.int64)
        cluster_sites = np.zeros((N_SITES, 2), dtype=np.int64)
        cluster_ids = np.zeros(N_SITES, dtype=np.int64)
        pocket[0], cluster_ids[0] = site, site
        cluster_sites[0] = n, m
        return pocket, cluster_sites, cluster_ids

    def expand(self, state, probability):
        pocket, cluster_sites, cluster_ids = self.start_cluster(BULK_SPIN,
                                                                BULK_BEAD)
        return expand_cluster(N_SPINS, N_BEADS, N_SLICES, TIME_COUPLING,
                              probability, probability, probability,
                              probability, 0, 0, state, pocket,
                              cluster_sites, 1, 1, BULK_SPIN, BULK_BEAD,
                              cluster_ids)

    def test_add_to_cluster(self):
        pocket, cluster_sites, cluster_ids = self.start_cluster(0, 0)
        site = NEW_BEAD + NEW_SPIN * N_SLICES
        cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
            add_to_cluster(NEW_SPIN, NEW_BEAD, 1, pocket, 1, cluster_sites,
                           site, cluster_ids)
        self.assertEqual((cluster_size, pocket_size), (2, 2))
        self.assertEqual(pocket[1], site)
        self.assertEqual(cluster_ids[1], site)
        self.assertEqual(list(cluster_sites[1]), [NEW_SPIN, NEW_BEAD])

    def test_expand_adds_all_aligned_neighbours(self):
        state = np.ones((N_SPINS, N_SLICES), dtype=np.bool_)
        cluster_size, pocket, pocket_size, cluster_sites, cluster_ids = \
            self.expand(state, ALWAYS)
        # two neighbours in space and two in imaginary time
        neighbours = {(BULK_SPIN - 1, BULK_BEAD), (BULK_SPIN + 1, BULK_BEAD),
                      (BULK_SPIN, BULK_BEAD - 1), (BULK_SPIN, BULK_BEAD + 1)}
        self.assertEqual(cluster_size, 1 + len(neighbours))
        self.assertEqual(pocket_size, len(neighbours))
        self.assertEqual({tuple(s) for s in cluster_sites[1:cluster_size]},
                         neighbours)

    def test_expand_adds_nothing(self):
        aligned = np.ones((N_SPINS, N_SLICES), dtype=np.bool_)
        cluster_size, _, pocket_size, _, _ = self.expand(aligned, NEVER)
        self.assertEqual((cluster_size, pocket_size), (1, 0))
        # opposite neighbours are never added, whatever the probability
        flipped = np.zeros((N_SPINS, N_SLICES), dtype=np.bool_)
        flipped[BULK_SPIN, BULK_BEAD] = True
        cluster_size, _, pocket_size, _, _ = self.expand(flipped, ALWAYS)
        self.assertEqual((cluster_size, pocket_size), (1, 0))


class TestScripts(unittest.TestCase):

    def test_missing_saveto(self):
        with self.assertRaises(ValueError):
            IsingMC_INT.main(['IsingMC_INT.py'] * INT_ARGUMENTS)
        with self.assertRaises(ValueError):
            IsingMC_RAT.main(['IsingMC_RAT.py'] * RAT_ARGUMENTS)


if __name__ == '__main__':
    unittest.main()

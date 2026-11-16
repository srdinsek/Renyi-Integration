"""
Physics test: Renyi entropy of two coupled harmonic oscillators.

The Monte Carlo result of AbInitioMC is compared with the exact result for
the same number of beads (see exact_oscillators.py). The exact result is
itself checked against the continuum formula of the PRB paper.

The Monte Carlo test runs N_POINTS processes in parallel and takes about a
minute per CPU. Run from the repository root with:
python -m unittest tests.test_oscillators
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'AbInitio'))
sys.path.insert(0, HERE)
from AbInitioMC import run_renyi, renyi_entropy
from exact_oscillators import exact_discrete_s2, exact_continuum_full_s2

M = 1.0
MASS = [[M / 2, 0], [0, M / 2]]
MU = [[1.0, 0.6], [0.6, 1.0]]
QUARTIC = [[0, 0], [0, 0]]
BETA = 3.0
N_BEADS = 6
PARTITION2 = 1  # entropy of the first oscillator
N_POINTS = 11
N_STEPS = 10 ** 6
START = 10 ** 4
SAMPLING_INTERVAL = 10
SAMPLING_OFFSET = 5
RANDOM_SEED = 11
N_SIGMA = 4  # allowed deviation in units of the error
MAX_RELATIVE_ERROR = 0.05  # the test is only meaningful if the error is small
CONTINUUM_BEADS = 256
CONTINUUM_TOLERANCE = 1e-5


def run_point(saveto, index, lam):
    zeta = BETA / N_BEADS
    run_renyi(BETA, zeta, (1 - lam) ** 3, 2, MASS, MU, QUARTIC, [0, 0], 0,
              PARTITION2, N_STEPS, np.sqrt(zeta / M), START, RANDOM_SEED,
              SAMPLING_OFFSET, SAMPLING_INTERVAL, 2, True, False, index,
              'oscillators', lam ** 3, 0, 'analytic', saveto=saveto)


class TestOscillators(unittest.TestCase):

    def test_exact_reference_against_continuum(self):
        '''Many beads: the discrete result approaches Eq. (C9) of the PRB.'''
        discrete = exact_discrete_s2(BETA, CONTINUUM_BEADS, MASS, MU, 2)
        self.assertAlmostEqual(discrete, exact_continuum_full_s2(BETA, M, MU),
                               delta=CONTINUUM_TOLERANCE)

    def test_monte_carlo_against_exact(self):
        path_points = np.linspace(0, 1, N_POINTS)
        with tempfile.TemporaryDirectory() as saveto:
            processes = [Process(target=run_point, args=(saveto, i, lam))
                         for i, lam in enumerate(path_points)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            entropy, error = renyi_entropy(saveto, 'oscillators',
                                           path_points,
                                           -3 * (1 - path_points) ** 2,
                                           3 * path_points ** 2, thin=1)
        exact = exact_discrete_s2(BETA, N_BEADS, MASS, MU, PARTITION2)
        print('\nS2 Monte Carlo', entropy, '+-', error, ' exact', exact)
        self.assertLess(error, MAX_RELATIVE_ERROR * exact)
        self.assertLess(abs(entropy - exact), N_SIGMA * error)


if __name__ == '__main__':
    unittest.main()

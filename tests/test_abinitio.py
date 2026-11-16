"""
Tests of the functions used inside the AbInitio algorithm.

Run from the repository root with: python -m unittest discover tests
"""

import os
import sys
import pickle
import tempfile
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'AbInitio'))
sys.path.insert(0, HERE)
from AbInitioMC import potential_energy, multi, initial_energy
from AbInitioMC import integrate_paper, renyi_entropy, run_renyi
from exact_oscillators import action_matrix

SEED = 0
ANALYTIC, REALISTIC = 0, 1
POSITION = np.array([0.3, -0.7])
MU = np.array([[1.0, 0.6], [0.6, 1.0]])
QUARTIC = np.array([[0.2, 0.1], [0.1, 0.3]])
NO_QUARTIC = np.zeros((2, 2))
GRID_SIZE, GRID_EDGE = 21, 1.0
BILINEAR = (0.5, -1.2, 0.8, 2.0)  # f(x, y) = a + b x + c y + d x y
OUTSIDE_POSITION = np.array([1.3, -1.6])  # outside the grid, extrapolated
WEIGHT = np.array([0.25, 1.5])
MASS = np.array([[0.5, 0.0], [0.0, 0.5]])
BETA, N_BEADS = 3.0, 6
LAMBDAS = (0.0, 0.3, 1.0)
DIMENSION = 2
TOLERANCE = 1e-12
ENERGY_TOLERANCE = 1e-9  # energies are sums of many terms of order 10
CUBIC = (1.0, -2.0, 0.5, 3.0)  # y = a + b x + c x^2 + d x^3
ODD_POINTS, EVEN_POINTS = 7, 4
SQUARE_EVEN_PAPER = 55 / 162  # old simps(even='avg') of x^2 on 4 points
PATH_POINTS = 5
T0_VALUE, T1_VALUE = 2.0, 3.0
N_SAMPLES, N_BLOCKS = 200, 4
SMALL_RUN = dict(n_steps=2000, start=100, sampling_interval=10)
SUFFIXES = ['_Ratio', '_Ratio_tot', '_T0_tot', '_T1_tot', '_K_tot',
            '_V_tot', '_path', '_x', '_V0_tot', '_V1_tot']


def bilinear(x, y):
    a, b, c, d = BILINEAR
    return a + b * x + c * y + d * x * y


def save_samples(folder, name, index, T0, T1):
    prefix = os.path.join(folder, name + 'INDEX_' + str(index))
    np.save(prefix + '_T0_tot', T0)
    np.save(prefix + '_T1_tot', T1)


def small_run(folder, index, lam, **kwargs):
    '''A very short analytic run of run_renyi.'''
    zeta = BETA / N_BEADS
    run_renyi(BETA, zeta, (1 - lam) ** 3, 2, MASS, MU, NO_QUARTIC, [0, 0],
              0, 1, SMALL_RUN['n_steps'], np.sqrt(zeta), SMALL_RUN['start'],
              SEED, 0, SMALL_RUN['sampling_interval'], DIMENSION, True,
              False, index, 'small', lam ** 3, 0, 'analytic', **kwargs)


class TestPotential(unittest.TestCase):

    def test_analytic(self):
        square = POSITION ** 2
        expected = POSITION @ MU @ POSITION / 2 + square @ QUARTIC @ square
        value = potential_energy(POSITION, MU, QUARTIC, ANALYTIC,
                                 np.zeros((2, 2)), np.zeros(2), 0.0)
        self.assertAlmostEqual(value, expected, delta=TOLERANCE)

    def test_interpolation_is_exact_for_bilinear(self):
        coor, step = np.linspace(-GRID_EDGE, GRID_EDGE, GRID_SIZE,
                                 retstep=True)
        X, Y = np.meshgrid(coor, coor, indexing='ij')
        potential = bilinear(X, Y)
        for position in (POSITION, OUTSIDE_POSITION):
            value = potential_energy(position, MU, NO_QUARTIC, REALISTIC,
                                     potential, coor, step ** 2)
            self.assertAlmostEqual(value, bilinear(*position),
                                   delta=TOLERANCE)

    def test_multi(self):
        self.assertAlmostEqual(multi(MU, POSITION, WEIGHT),
                               POSITION @ MU @ (POSITION * WEIGHT),
                               delta=TOLERANCE)


class TestEnergy(unittest.TestCase):

    def test_initial_energy_equals_quadratic_form(self):
        '''The energy of the code is x^T A x of the exact oscillators.'''
        zeta = BETA / N_BEADS
        rng = np.random.default_rng(SEED)
        for partition2 in (1, 2):
            for lam in LAMBDAS:
                c1, c2 = (1 - lam) ** 3, lam ** 3
                lam1 = np.ones(DIMENSION) / 2
                lam1[:partition2] *= c1 * 2
                lam2 = np.ones(DIMENSION) / 2
                lam2[:partition2] *= c2 * 2
                path = rng.normal(0, 1, (DIMENSION, 2 * N_BEADS))
                empty = np.zeros(DIMENSION)
                V, K1, K2, E01, E02 = initial_energy(path, empty.copy(),
                    empty.copy(), empty.copy(), empty.copy(), N_BEADS, 0,
                    partition2, MASS / zeta, MU * zeta, NO_QUARTIC, ANALYTIC,
                    np.zeros((2, 2)), np.zeros(2), 0.0, lam1, lam2, c1, c2,
                    False)
                A = action_matrix(BETA, N_BEADS, MASS, MU, partition2, c1, c2)
                x = path.ravel()
                self.assertAlmostEqual(E01, x @ A @ x, delta=ENERGY_TOLERANCE)


class TestPostProcessing(unittest.TestCase):

    def test_integrate_paper_odd(self):
        x = np.linspace(0, 1, ODD_POINTS)
        a, b, c, d = CUBIC
        y = a + b * x + c * x ** 2 + d * x ** 3
        self.assertAlmostEqual(integrate_paper(y, x),
                               a + b / 2 + c / 3 + d / 4, delta=TOLERANCE)

    def test_integrate_paper_even(self):
        x = np.linspace(0, 1, EVEN_POINTS)
        self.assertAlmostEqual(integrate_paper(x ** 2, x), SQUARE_EVEN_PAPER,
                               delta=TOLERANCE)

    def test_renyi_entropy_constant_samples(self):
        lam = np.linspace(0, 1, PATH_POINTS)
        dlam1, dlam2 = -3 * (1 - lam) ** 2, 3 * lam ** 2
        with tempfile.TemporaryDirectory() as folder:
            for i in range(PATH_POINTS):
                save_samples(folder, 'c', i, np.full(N_SAMPLES, T0_VALUE),
                             np.full(N_SAMPLES, T1_VALUE))
            entropy, error = renyi_entropy(folder, 'c', lam, dlam1, dlam2,
                                           thin=1)
        # integral of T0 dlam1 + T1 dlam2 from 0 to 1 is T1 - T0
        self.assertAlmostEqual(entropy, T1_VALUE - T0_VALUE, delta=TOLERANCE)
        self.assertAlmostEqual(error, 0, delta=TOLERANCE)

    def test_renyi_entropy_error_of_the_mean(self):
        # block b of the samples has the value b, the integrand is T0
        lam = np.linspace(0, 1, PATH_POINTS)
        blocks = np.arange(N_BLOCKS, dtype=np.float64)
        samples = np.repeat(blocks, N_SAMPLES // N_BLOCKS)
        with tempfile.TemporaryDirectory() as folder:
            for i in range(PATH_POINTS):
                save_samples(folder, 'b', i, samples, np.zeros(N_SAMPLES))
            entropy, error = renyi_entropy(folder, 'b', lam,
                                           np.ones(PATH_POINTS),
                                           np.zeros(PATH_POINTS), thin=1,
                                           n_blocks=N_BLOCKS)
        self.assertAlmostEqual(entropy, np.mean(blocks), delta=TOLERANCE)
        self.assertAlmostEqual(error, np.std(blocks) / np.sqrt(N_BLOCKS),
                               delta=TOLERANCE)


class TestRun(unittest.TestCase):

    def test_missing_saveto(self):
        with self.assertRaises(TypeError):
            small_run('.', 0, 0.5)

    def test_written_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, lam in enumerate(LAMBDAS):
                small_run(folder, index, lam, saveto=folder)
            for index, lam in enumerate(LAMBDAS):
                prefix = os.path.join(folder, 'smallINDEX_' + str(index))
                for suffix in SUFFIXES:
                    self.assertTrue(os.path.exists(prefix + suffix + '.npy'))
                with open(prefix + '_args.p', 'rb') as fp:
                    parameters = pickle.load(fp)
                self.assertEqual(parameters['lam1'], (1 - lam) ** 3)
                self.assertEqual(parameters['lam2'], lam ** 3)


if __name__ == '__main__':
    unittest.main()

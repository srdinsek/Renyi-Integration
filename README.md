# Renyi-Integration

Calculations reported in the paper ["Quantum Rényi entropy by optimal thermodynamic integration paths"](https://doi.org/10.1103/PhysRevResearch.4.L032002) (M. Srdinšek, M. Casula, R. Vuilleumier, *Phys. Rev. Research* **4**, L032002 (2022)) were performed using the routines in this repository. The folder **Ising** contains programs for the evaluation of the Rényi entropy in the quantum Ising model, and the folder **AbInitio** for the evaluation in an *ab initio* model with an arbitrary potential. Only the second-order Rényi entropy is supported.

## Installation

The code needs Python 3 with numpy, numba and scipy. In a fresh virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ising model

The algorithm defined in **IsingMC_main_cluster.py** was used to compare thermodynamic integration in Ising model to the evaluation of the *SWAP* operator. Different procedures can be used by running either **IsingMC_RAT.py** for *SWAP* or **IsingMC_INT.py** for thermodynamic integration. The two programs are run by calling

```
python3 IsingMC_RAT.py beta zeta n_spins r_min r_max n_fields n_steps n_points saveto

python3 IsingMC_INT.py beta zeta n_spins r_min r_max n_fields n_steps subsystem n_points saveto
```

**Each program requires `n_points` processors.**

The parameters define:
- `beta` - inverse temperature
- `zeta` - imaginary-time step $\zeta = \beta / M$, where $M$ is the number of beads per replica. The program sets $M$ = `int(beta / zeta)` and then resets $\zeta = \beta / M$, so that $M$ is an integer.
- `n_spins` - number of spins
- `r_min, r_max, n_fields` - specifies the magnetic fields used. Means: `numpy.linspace(r_min, r_max, n_fields)`
- `n_steps` - number of steps in [Wolff cluster algorithm](https://en.wikipedia.org/wiki/Wolff_algorithm). Because clusters are larger at small `r`, the program multiplies `n_steps` by 60 for `r > 3` and by 10 for `1 < r <= 3`, and records observables every 10 steps for `r < 1`.
- `subsystem` ('full' or 'half') - specifies if in thermodynamic integration full entropy or entanglement entropy of the half should be evaluated
- `n_points` - number of integration steps (thermodynamic integration, there is no actual integration in Monte Carlo algorithm) in thermodynamic integration and number of intermediate states when measuring *SWAP*. Equals the number of processors used.
- `saveto` - folder where all results are saved

In the files there are additional parameters that can be modified. 

    start = 10 ** 4 # when to start sampling the observable
    sampling_interval = 100 # every sampling_interval step observables are recorded
    n_walkers = 5 # number of walkers - parallel runs of the same system

## Ab initio

The algorithm defined in **AbInitioMC.py** can be used to compute Rényi entanglement entropy in *ab initio* analytic potentials. Realistic potentials are implemented only for 2D, but the program can be easily extended to higher dimension. The program can be run by calling

    python3 Example.py

**The program requires `n_points` processors.**

At the end, **Example.py** integrates the results along the path with `renyi_entropy()` (defined in **AbInitioMC.py**), prints the second-order Rényi entropy with its error, and saves both (in nats) to `name + '_entropy.npy'`. The error is the error of the mean from 20 blocks; the figures of the paper show 3 times this value.

All the necessary parameters are defined in **Example.py**. **Example.py** is set for the evaluation of Rényi entanglement entropy of one Hydrogen in realistic 2D potential. Potential is defined in **AbInitioMC.py** and can be modified to suit ones needs.

Parameters defined below can be modified, others are redundant, or shouldn't be modified.
- `beta` - inverse temperature
- `design` - requested imaginary-time step $\zeta = \beta / M$, where $M$ is the number of beads per replica. **Example.py** chooses $M$ from `int(beta / design)`, limited to between 6 and 300, and then uses $\zeta = \beta / M$.
- `name` - head of the files with results
- `saveto` - folder where all results are saved
- `index` - number that further distinguishes the files with results
- `n_points` - number of integration steps (thermodynamic integration, there is no actual integration in Monte Carlo algorithm). Equals the number of processors used.
- `m` - mass of the quantum particle 
- `lamda` - parameter matrix defining the potential **used only if potential_type='analytic'**
- `gamma` - parameter matrix defining the potential **used only if potential_type='analytic'**
- `mass` - matrix describing the interaction in *imaginary-time* direction. For 2D, it is simply [[m / 2, 0], [0, m / 2]].
- `shift` - shift of the initial state
- `partition1` - number of joint copies starting from particle 0. Zero means there are none.
- `partition2` - specifies the trial size starting from particle 0.
- `n_steps` - number of steps in [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
- `start` - when to start sampling the observable
- `reflection_interval` - every reflection_interval step, move with reflecting the positions of the copies is proposed
- `sampling_interval` - every sampling_interval step observables are recorded
- `dimension` - how many particles there are
- `random_seed` - seed for random numbers
- `potential_type` ('analytic' or 'realistic') - specifies the type of potential.
- `real_index` - which grid of **potentials.npy** (and row of **distances.npy**) is used if potential_type='realistic'

If `partition1` is nonzero, then the trial size is `partition2 - partition1`. Trial copies are the ones that we modify during the thermodynamic integration. For example:
1. function with parameters `partition2`, `partition1=0` simulates split ensemble and modifies the subsystem spanning from particle 0 to particle (partition2 - 1).
2. The function with parameters `partition2=0`, `partition1` simulates ensemble, where copies spanning from particle 0 to particle (partition1 - 1) are joined.

### Example potential

Any 2D potential on a square grid can be used by replacing **potentials.npy** and **distances.npy**. The files in this repository are the ones used in the paper:
- **potentials.npy** - 38 grids of 30 × 30 CCSD(T) total energies (Hartree, computed with PySCF) of the formic-acid dimer. The two hydrogens of the hydrogen bonds are moved along their O···O bonds, from 0.5 Å to (d - 0.5) Å; the other atoms are kept at the equilibrium geometry optimised with ORCA, except that one formic-acid molecule is moved rigidly along the C···C axis to change the O···O distances.
- **distances.npy** - the two O···O distances d (Å) of each grid. It has 40 rows: row i belongs to grid i, and the last two rows have no grid, so use non-negative values of `real_index`.

The paper used `real_index` from 13 to 29.

## Citation

If you use this code, please cite:

```bibtex
@article{Srdinsek2022,
  title   = {Quantum R\'enyi entropy by optimal thermodynamic integration paths},
  author  = {Srdin\v{s}ek, Miha and Casula, Michele and Vuilleumier, Rodolphe},
  journal = {Phys. Rev. Research},
  volume  = {4},
  issue   = {3},
  pages   = {L032002},
  year    = {2022},
  doi     = {10.1103/PhysRevResearch.4.L032002},
  url     = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.L032002}
}
```

## License

[MIT](LICENSE)

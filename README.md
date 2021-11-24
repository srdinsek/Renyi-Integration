# Renyi-Integration

Calculations reported in the paper _________ were performed using the rutines in this repository. First map contains programs for evaluation of Rényi entropy in the Ising model and the second one for the evaluation in *ab initio* model with arbitrary potential. 

## Ising model

The algorithm defined in **IsingMC_main_cluster.py** was used to compare thermodynamic integration in Ising model to the evaluation of the *SWAP* operator. Different procedures can be used by running either **IsingMC_RAT.py** for *SWAP* or **IsingMC_INT.py** for thermodynamic integration. `SAVETO` at the begining of **IsingMC_RAT.py** and **IsingMC_INT.py** specifies where final results (after post processing) should be stored, but notice that there is another one in **IsingMC_main_cluster.py**, that specifies where intermediate results should be stored. The two programs are run by calling

```
python3 IsingMC_RAT.py beta zeta N r_min r_max r_N NUMBER_OF_STEPS LEN

python3 IsingMC_INT.py beta zeta N r_min r_max r_N NUMBER_OF_STEPS GESLO LEN
```

**Each program requires `LEN` pocessors.**

The parameters define:
- `beta` - inverse temperature
- `zeta` - $\zeta$ that we want (but can be modified, due to the min and max number of beads)
- `N` - number of spins
- `r_min, r_max, r_N` - specifies the magnetic fields used. Means: `numpy.linspace(r_min, r_max, r_N)`
- `NUMBER_OF_STEPS` - number of steps in [Wolff cluster algorithm](https://en.wikipedia.org/wiki/Wolff_algorithm)
- `GESLO` ('full' or 'half')- specifies if in thermodynamic integration full entropy or entanglement entropy of the half should be evaluated
- `LEN` - number of integration steps (thermodynamic integration, there is no actuall integration in Monte Carlo algorithm) in thermodynymic integration and number of intermediate states when measuring *SWAP*. Equals the number of processors used.

In the files there are aditional parameters that can be modified. 

    start = 10 ** 4 # when to start sampling the observable
    SAMPLING_NUMBER = 100 # every SAMPLING_NUMBER step observables are recorded
    N_WALKERS = 5 # number of walkers - paralle runs of the same system

## Ab initio

The algorithm defined in **AbInitioMC.py** can be used to compute Rényi entanglement entropy in *ab initio* analytic potentials. Realistic potentials are implemented only for 2D, but the program can be easily extended to higher dimension. Program can be run by calling.

    python3 Example.py

**Program requires `LEN` pocessors.**

All the necesary parameters are defined in **Example.py**. **Example.py** is set for the evaluation of Rényi entanglement entopy of one Hydrogen in realistic 2D potential. To change the directory on which results will be saved, modify `SAVETO` at the begining of **AbInitioMC.py**. Potential is defined in **AbInitioMC.py** and can be modified to suit ones needs.

Parameters defined bellow can be modified, others are redundant, or shouldn't be modified.
- `beta` - inverse temperature
- `design` - $\zeta$ that we want (but can be modified, due to the min and max number of beads)
- `NAME` - head of the files with results
- `index` - numer that further distinguishes he files with results
- `LEN` - number of integration steps (thermodynamic integration, there is no actuall integration in Monte Carlo algorithm). Equals the number of processors used.
- `m` - mass of the quantum particle 
- `lamda` - parameter matrix defining the potential **used only if POTENTIAL_TYPE='analytic'**
- `gamma` - parameter matrix defining the potential **used only if POTENTIAL_TYPE='analytic'**
- `mass` - matrix describing the interaction in *imaginary-time* direction. For 2D, it is simply [[m / 2, 0], [0, m / 2]].
- `shift` - shift of the initial state
- `partition1` - number of joint copies starting from particle 0. Zero means there are none.
- `partition2` - specifies the trial size starting from particle 0.
- `NUMBER_OF_STEPS` - number of steps in [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
- `start` - when to start sampling the observable
- `class_move_number` - every class_move_number step, move with reflecting the positions of the copies is proposed
- `SAMPLING_NUMBER` - every SAMPLING_NUMBER stepw observables are recorded
- `dimension` - how many particles there are
- `SEED` - seed for random numbers
- `POTENTIAL_TYPE` ('analytic' or 'realistic') - specifies the type of potential.

If `partition1` is nonzero, then the trial size is `partition2 - partition1`. Trial copies are the ones that we modify during the thermodynamic integration. For example:
1. function with parameters `partition2`, `partition1=0` simulates split ensemble and modifies the subsystem spaning from particle 0 to particle (partition2 - 1).
2. The function with parameters `partition2=0`, `partition1` simulates ensemlbe, where copies spanning from particle 0 to particle (partition1 - 1) are joined. 




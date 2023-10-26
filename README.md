# Abstract

For a description of the problem check the report PIMLB_project_report.pdf of this repo.

# Content

This repository contains:

- A report of all the activities conducted inside the MaLGa PiMLB unit --> PiMLB_project_report.pdf
- A folder "cuda_kernels" containing the implementations of the single cuda kernels and a document explaining the computations of the kernels.

# Instructions to run the CPU code (not included in this repository)

## Prerequisites

- Install the [GSL library](https://www.gnu.org/software/gsl/)
- Install [zlib](https://zlib.net/) (only for execution on local machine[*](https://stackoverflow.com/questions/10440113/simple-way-to-unzip-a-zip-file-using-zlib))

## Compilation

First, in Makefile, set the INC and LIB variables to the paths where you have GSL installed (the folder "include"
and "lib" of the GSL absolute path, respectively)

Example:

```
INC =I/usr/local/include/gsl
LIBS =L/usr/local/include/gsl
```

Open `pomdp_sniff.cpp` and edit the following parameters:

```
int collectmethod = 2; // 0 Random collect, 1 FSVI Collect, 2 FSVI Original, 3 PBVI Collect
```
- `collectmethod` denotes the algorithm used for the belief collection at training time

```
int backupmethod = 1; // 0 Partial backup, 1 Full backup
```
- `backupmethod` denotes the algorithm used for the backup phase at training time

`collectmethod = 2` and `backupmethod = 1` are the default choices used throughout the experiments.

Then, run

```
make
```

## Run

### Local machine

#### Training + test with the same starting position

Open `pomdp_sniff.cpp` and edit the following parameters:

Open `launch_multiple_exp.sh` and edit the parameters as follows. Notice that parameters with arguments specified
inside brackets denote the possibility of launching multiple experiments at once (clarified in the description).

- `OUT_DIR="./exp/test"` is the path of the output folder
- `Numtrials=1` (do not modify)
- `Maxsteps_test=500` is the maximum number of steps in the test phase
- `Infotaxis=0` flag to activate infotaxis (do not modify)
- `Seed_test=(1 2 3 4 5)` seed at test time. Multiple seeds correspond to a number of repetition of the experiments (e.g., 3 seeds = 3 experiments)
- `Seed_train=(6 7 8 9 10)` seed at training time. The number of seeds at training time must be equal to the number of seeds at test time (e.g., 3 seeds at test time require 3 seeds at training time)
- `Gamma=(0.99)` parameter $\gamma$ in the Bellman equation 
- `Numstates_per_length=(20)` number of cells $\lambda$ per unit length of the environment
- `Pomdpsolve_numiter=(320)` number of iterations of the POMDP solver algorithm
- `Pomdpsolve_numpoints=(1)` number of belief points considered at training time. Valid only with `collectmethod = 1` (FSVI Collect) or `collectmethod = 3` (PBVI).
- `X_start=(8.00)` is the starting $x$ coordinate of the agent in the environment
- `Y_start=(0.00)` is the starting $y$ coordinate of the agent in the environment

Bounding box of the training prior (multiple coordinates correspond to different experiments with different training prior. The number of coordinates must be the same for each of the following parameters):
- `X_min=(0.00 -0.50 -0.80 -1.50 -1.90)` 
- `X_max=(8.50 9.00 9.30 9.50 9.90)`
- `Y_min=(-0.30 -0.50 -0.60 -0.70 -0.90)`
- `Y_max=(0.30 0.50 0.60 0.70 0.90)`

Bounding box of the test prior (multiple coordinates correspond to different experiments with different training prior. The number of coordinates must be the same for each of the following parameters):
- `X_min_test=(0.00)`
- `X_max_test=(8.50)`
- `Y_min_test=(-0.30)`
- `Y_max_test=(0.30)`


Then, from the main directory of the code run
```
./launch_multiple_exp.sh
```


#### Test with different prior and/or starting positions

Once you have solved the POMDP, you can test your solution on a different prior and/or a different starting position.
Open `launch_multiple_exp_spos.sh` and edit the parameters as follows.

- `OUT_DIR="./exp/diag/niter_320"` is the path of the output folder. Make sure it contains the file `alphavecs.dat` (solution of the POMDP).
- `X_start=(9.00 4.00 -1.50)` is the $x$ coordinate of the agent starting position (multiple coordinates correspond to different experiments with different starting position)
- `Y_start=(0.90 -0.80 0.70)` is the $y$ coordinate of the agent starting position (make sure the number of $y$ coordinates match the number of $x$ coordinates). 
- `train_xstart=8.00` is the agent starting $x$ coordinate at training time
- `train_ystart=0.00` is the agent starting $y$ coordinate at training time

Make sure all the other parameters coorespond to the ones used at training time.



### UNice CPU Cluster

#### Training + test with the same starting position

Open `launch_cluster.sh` and edit the parameters as described in the analogous paragraph of the previous section (Local machine --> Training + test with the same starting position)

#### Test with different prior and/or starting positions

Open `launch_spos_cluster.sh` and edit the parameters as described in the analogous paragraph of the previous section (Local machine --> Test with different prior and/or starting positions)

### GPU Cluster
Here instruction on how to run on the PiMLB GPU cluster

## Output

In the following, `{xstart}, {ystart}` denote the agent starting position, `{xmin_test}, {xmax_test}, {ymin_test}, {ymax_test}` denote the bounding box of the prior at test time.

#### Training + test with the same starting position

In the path specified as `OUT_DIR` the following files will be created:

- `alphavecs.dat` contains the alpha vectors computed at training time
- `beliefvecs_test_{xstart}_{ystart}_bt_{xmin_test}_{xmax_test}_{ymin_test}_{ymax_test}.zip` contains the beliefs encountered at test time
- `beliefvecs_train.zip` contains the beliefs collected at training time
- `log.txt` is a log file
- `starting_points.dat` contains the starting points used at training time (with the FSVI algorithm)
- `stderr.txt` is the standard error
- `test_results_{xstart}_{ystart}_bt_{xmin_test}_{xmax_test}_{ymin_test}_{ymax_test}.dat` contains the results of the test phase
- `traj_{xstart}_{ystart}_bt_{xmin_test}_{xmax_test}_{ymin_test}_{ymax_test}.dat` contains the trajectory of the agent at test time


#### Test with different starting positions

In the path specified as `OUT_DIR` the following files will be created:

- `beliefvecs_test_{xstart}_{ystart}_bt_{xmin_test}_{xmax_test}_{ymin_test}_{ymax_test}.zip` contains the beliefs encountered at test time
- `traj_{xstart}_{ystart}_bt_{xmin_test}_{xmax_test}_{ymin_test}_{ymax_test}.dat` contains the trajectory of the agent at test time
- `stderr_eval.txt` is the standard error


## Python utils

- `plot_st_points.py` can be used to plot the starting points used at training time with the FSVI algorithm
- `plot_statistics_rev.py` can be used to plot statistics of the experiments (e.g., additional time to target, JS score, cast length)
- `plot_traj_belief_new.py`cam be used to plot trajectories
- `plot_beliefs.py` can be used to plot evolution of beliefs at test time or beliefs at training time

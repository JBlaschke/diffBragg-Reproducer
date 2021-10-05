# Derek's MLX5 Reproducer

The MLX5 error occurs (pretty consistently) on Perlmutter's GPU nodes.

## Installation

Two variants can be installed:
1. Using NERSC's Conda (via the `python` module)
2. Using Miniconda

If you choose to use (2) then you need to source the `env.sh` before running.
If you choose (1) then you need to run `module load PrgEnv-gnu cuda cpe-cuda
python` instead. c.f. bullet 2 in the section on `Running`.

### Using NERSC's Conda

```bash
./setup-noconda.sh $YOUR_LOCAL_CONDA_PREFIX
```
This will create a conda environment in the locaction `YOUR_LOCAL_CONDA_PREFIX`
conda prefix. Re-running will **not** delete this environment.

### Miniconda Variant

```bash
./setup.sh
```
**Note:** this will erase any previous conda installations, **including** the
conda environment

## Running

1. Grab 77 interactive nodes:
```bash
salloc -C gpu -n 308 -c 2 -t 60 --gpus-per-task 1 -A <account>_g -q early_science
```
where `<account>` is your Perlmutter Early-Science account.
2. Run the reproducer -- depending on your variant activating the environment
different:

### Using NERSC's Conda

If you installed using `setup-noconda.sh $YOUR_LOCAL_CONDA_PREFIX`
```bash
module load PrgEnv-gnu cuda cpe-cuda python
source activate $YOUR_LOCAL_CONDA_PREFIX
srun -N77 -n308 -c2 python -X faulthandler pmUsrIssue5.py
```


### Miniconda Variant

If you installed using `setup.sh`
```bash
source env.sh
conda activate diffbrag2
srun -N77 -n308 -c2 python -X faulthandler pmUsrIssue5.py
```
**NOTE:** the `env.sh` script loads all necessary modules on Perlmutter and
puts this repo's conda into the path

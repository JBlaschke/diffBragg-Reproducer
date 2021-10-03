# Derek's MLX5 Reproducer

The MLX5 error occurs (pretty consistently) on Perlmutter's GPU nodes.

## Installation

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
2. Run the reproducer:
```bash
source env.sh
srun -N77 -n308 -c2 python -X faulthandler pmUsrIssue5.py
```

**NOTE:** the `env.sh` script loads all necessary modules on Perlmutter and
puts this repo's conda into the path

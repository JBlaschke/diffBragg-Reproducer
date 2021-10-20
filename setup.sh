#!/bin/bash

set -e

env=$(pwd)/diffbragg

# Load modules
module load PrgEnv-gnu python

# Set up diffBragg's conda environtment
conda create --prefix=$env python=3.7 -y
source activate $env

MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
conda install -c conda-forge cctbx-base -y
conda install pandas -y

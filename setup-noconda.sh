#!/bin/bash

# Load modules
module load PrgEnv-gnu cuda cpe-cuda python

# Set up diffBragg's conda environtment
conda create --prefix=$1 python=3.7 -y
source activate $1

MPICC="cc -shared -target-accel=nvidia80 -lmlx5" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
conda install -c conda-forge cctbx-base -y
conda install pandas -y

#!/bin/bash
module load PrgEnv-gnu python cuda cpe-cuda

conda create -n diffbrag2 python=3.7 -y
conda activate diffbrag2

MPICC="cc -shared -target-accel=nvidia80 -lmlx5" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
conda install -c conda-forge cctbx-base -y
conda install pandas -y


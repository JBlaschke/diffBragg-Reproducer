#!/bin/bash

source $(readlink -f $(dirname "${BASH_SOURCE[0]}"))/env.sh

rm -rf $CONDA_ROOT


# Get and run the miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -p).sh -O conda-installer.sh
bash ./conda-installer.sh -b -p $CONDA_ROOT
rm conda-installer.sh

source ${CONDA_ROOT}/etc/profile.d/conda.sh

# Set up diffBragg's conda environtment
conda create -n diffbrag2 python=3.7 -y
conda activate diffbrag2

MPICC="cc -shared -target-accel=nvidia80 -lmlx5" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
conda install -c conda-forge cctbx-base -y
conda install pandas -y

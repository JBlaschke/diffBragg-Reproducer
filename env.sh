# Load modules
module load PrgEnv-gnu cuda cpe-cuda

# Set Environment
root_dir=$(readlink -f $(dirname "${BASH_SOURCE[0]}"))

export ROOT_DIR=${root_dir}
export CONDA_ROOT=${root_dir}/conda

if [[ -e ${CONDA_ROOT}/etc/profile.d/conda.sh ]]
then
    source ${CONDA_ROOT}/etc/profile.d/conda.sh 
fi

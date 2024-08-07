#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --partition=gpu          # gpu partition
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)


# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module purge

eval "$(conda shell.bash hook)"

conda activate torch-env

kernprof -o ${SLURM_JOBID}.lprof -l mnist_classify.py --epochs=3

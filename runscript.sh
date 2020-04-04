#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t 0-08:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ch_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ch_errors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc01 # Load module
source activate openaigym            # Switch to openaigym conda environment
python3 ./dqn_original.py --batchmode # Run code

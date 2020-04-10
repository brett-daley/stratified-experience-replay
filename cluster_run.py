import argparse
import subprocess
import os.path

######### RUN PARAMETERS #########
env_grid = ['atlantis', 'breakout', 'beam_rider', 'centipede', 'name_this_game', 'pong', 'road_runner', 'seaquest']
n_grid = [1, 2, 3, 4]
seed_grid = [0]
batch_flag_grid = [True]

# CAUTION: Changes in timesteps will NOT be reflected in output/err file names
timesteps = 3_000_000
##################################


def check_file_exists(env, n, seed, batch_flag):
    """
    Returns True if out or err file already exists, and False otherwise
    """
    env_no_underscore = env.replace("_", "")
    if os.path.exists(f'env-{env_no_underscore}_n-{n}_batchmode-{batch_flag}_seed-{seed}.txt') \
            or os.path.exists(f'env-{env_no_underscore}_n-{n}_batchmode-{batch_flag}_seed-{seed}.err.txt'):
        return True

    else:
        return False


def make_sh_file(env, n, timesteps, seed, batch_flag):
    """
    Populates 'runscript.sh' file to run 'dqn_original.py' file
    on cluster's GPU partition for 4 hours with 1 node, 1 core, and 32GB memory
    """

    # Reformat env name for output file name
    env_no_underscore = env.replace("_", "")

    with open('runscript.sh', 'w+') as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t 0-04:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o env-{env_no_underscore}_n-{n}_batchmode-{batch_flag}_seed-{seed}.txt  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e env-{env_no_underscore}_n-{n}_batchmode-{batch_flag}_seed-{seed}.err.txt  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/5.0.1-fasrc01  # Load module
source activate openaigym  # Switch to openaigym conda environment
python3 dqn_original.py --env {env} -n {n} --batchmode {batch_flag} --timesteps {timesteps} --seed {seed}  # Run code
"""
        )
        f.close()


def submit_job():
    """
    Dispatches runscript.sh to SLURM
    """
    subprocess.call(['sbatch', 'runscript.sh'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--go', action='store_true',
                        help='(flag) Submits jobs to cluster if present. Default disabled')
    args = parser.parse_args()

    for env in env_grid:
        for n in n_grid:
            for batch_flag in batch_flag_grid:
                for seed in seed_grid:

                    # If file for a configuration exists, skip over that configuration
                    if check_file_exists(env, n, seed, batch_flag) is True:
                        print(f'File already exists for env: {env}, n: {n}, batch_flag: {batch_flag}, seed: {seed}')
                        print("Skipping to next configuration")
                        continue

                    # Otherwise, generate and run script on cluster

                    # Populates 'runscript.sh' file to run 'dqn_original.py' file
                    # on cluster's GPU partition for 4 hours with 1 node, 1 core, and 32GB memory
                    make_sh_file(env, n, timesteps, seed, batch_flag)

                    # Dispatches 'runscript.sh' to SLURM if '--go' flag was specified in CLI
                    if args.go:
                        submit_job()


if __name__ == '__main__':
    main()

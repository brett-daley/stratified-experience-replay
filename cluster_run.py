import argparse
import subprocess
import os.path

######### RUN PARAMETERS #########
env_grid = ['atlantis', 'breakout', 'beam_rider', 'centipede', 'name_this_game', 'pong', 'road_runner', 'seaquest']
n_grid = [1, 2, 3, 4]
seed_grid = [0]
batch_flag_grid = [False, True]

# CAUTION: Changes in timesteps will NOT be reflected in output/err file names
timesteps = 3_000_000
##################################


def dispatch(out_file, err_file, cmd, go):
    """
    Populates 'runscript.sh' file to run 'dqn_original.py' file
    on cluster's GPU partition for 4 hours with 1 node, 1 core, and 32GB memory
    """
    with open('runscript.sh', 'w+') as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t 0-04:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o {out_file}  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e {err_file}  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/5.0.1-fasrc01  # Load module
source activate openaigym  # Switch to openaigym conda environment
{cmd}  # Run code
"""
        )
    if go:
        subprocess.call(['sbatch', 'runscript.sh'])


def print_red(string):
    print('\033[1;31;40m' + string + '\033[0;37;40m')

def print_yellow(string):
    print('\033[1;33;40m' + string + '\033[0;37;40m')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--go', action='store_true',
                        help='(flag) Submits jobs to cluster if present. Default disabled')
    args = parser.parse_args()

    for env in env_grid:
        for n in n_grid:
            for batch_flag in batch_flag_grid:
                for seed in seed_grid:
                    # Generate file paths and executable command
                    env_no_underscore = env.replace("_", "")  # Reformat env name for output file name
                    basename = f'env-{env_no_underscore}_n-{n}_batchmode-{batch_flag}_seed-{seed}'
                    out_file = basename + '.txt'
                    err_file = basename + '.err.txt'
                    cmd = f'python3 dqn_original.py --env {env} -n {n} --batchmode {batch_flag} --timesteps {timesteps} --seed {seed}'

                    # If file for a configuration exists, skip over that configuration
                    if os.path.exists(out_file) or os.path.exists(err_file):
                        print_red(f'{basename} (already exists; skipping)')
                        continue

                    # Otherwise, generate and run script on cluster
                    # Populates 'runscript.sh' file to run 'dqn_original.py' file
                    # on cluster's GPU partition for 4 hours with 1 node, 1 core, and 32GB memory
                    # Dispatches 'runscript.sh' to SLURM if '--go' flag was specified in CLI
                    print(basename)
                    dispatch(out_file, err_file, cmd, args.go)

    if not args.go:
        print_yellow('''
*** This was just a test! No jobs were actually dispatched.
*** If the output looks correct, re-run with the "--go" argument.''')
        print(flush=True)


if __name__ == '__main__':
    main()

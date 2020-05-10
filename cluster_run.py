import argparse
import subprocess
import os.path
import math

######### RUN PARAMETERS #########
env_grid = ['beam_rider', 'breakout', 'pong', 'seaquest', 'space_invaders']
n_grid = [1, 3, 5, 7]  # n-step learning
m_grid = [0, 2, 3, 4]  # m-strap learning (0 means disabled)
seed_grid = range(1)

# CAUTION: Changes in timesteps will NOT be reflected in output/err file names
timesteps = 5_000_000
##################################


def dispatch(out_file, err_file, cmd, max_hours, go):
    """
    Populates 'runscript.sh' file to run 'dqn_original.py' file
    on cluster's GPU partition for 'max_hours' hours with 1 node, 1 core, and 32GB memory
    """
    with open('runscript.sh', 'w+') as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t {format_time(max_hours)}           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o {out_file}  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e {err_file}  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/5.0.1-fasrc01  # Load module
module load cudnn/7.6.5.32_cuda10.1-fasrc01
source activate openaigym  # Switch to openaigym conda environment
{cmd}  # Run code
"""
        )
    if go:
        subprocess.call(['sbatch', 'runscript.sh'])


def format_time(total_hours):
    '''Converts hours to D-HH:MM format.'''
    days = total_hours // 24
    frac_hour, hours = math.modf(total_hours % 24)
    minutes = math.ceil(frac_hour * 60.0)
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours == 24:
        hours = 0
        days += 1
    return f'{int(days)}-{int(hours):02d}:{int(minutes):02d}'


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
            for m in m_grid:
                for seed in seed_grid:
                    # Generate file paths and executable command
                    env_no_underscore = env.replace("_", "")  # Reformat env name for output file name
                    basename = f'env-{env_no_underscore}_n-{n}_m-{m}_seed-{seed}'
                    out_file = basename + '.txt'
                    err_file = basename + '.err.txt'
                    cmd = f'python3 dqn_original.py --env {env} -n {n} -m {m} --timesteps {timesteps} --seed {seed}'

                    # If file for a configuration exists, skip over that configuration
                    if os.path.exists(out_file) or os.path.exists(err_file):
                        print_red(f'{basename} (already exists; skipping)')
                        continue

                    # Otherwise, generate and run script on cluster
                    # Populates 'runscript.sh' file to run 'dqn_original.py' file
                    # on cluster's GPU partition with 1 node, 1 core, and 32GB memory
                    # Dispatches 'runscript.sh' to SLURM if '--go' flag was specified in CLI
                    print(basename)
                    # Timing heuristic: 3 hours per 2500 minibatches/epoch, plus an extra hour
                    # hours = (3*n + 1) if batch_flag else (3*m/2500 + 1)
                    dispatch(out_file, err_file, cmd, 4, args.go)

    if not args.go:
        print_yellow('''
*** This was just a test! No jobs were actually dispatched.
*** If the output looks correct, re-run with the "--go" argument.''')
        print(flush=True)


if __name__ == '__main__':
    main()

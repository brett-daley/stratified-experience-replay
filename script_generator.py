import argparse


def main(script_name, runfile, env, n, timesteps, seed, batch_flag):
    if batch_flag:
        batch = '--batchmode'
    else:
        batch = ""
    with open(f'{script_name}', 'w+') as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t 0-14:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o env-{env}_n-{n}_tsteps-{timesteps}_seed-{seed}_range-2500_batchmode-{batch_flag}_jobID-%j_out.txt  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e env-{env}_n-{n}_tsteps-{timesteps}_seed-{seed}_range-2500_batchmode-{batch_flag}_jobID-%j_err.txt  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/5.0.1-fasrc01  # Load module
source activate openaigym  # Switch to openaigym conda environment
python3 ./{runfile} --env {env} --timesteps {timesteps} --seed {seed} {batch}  # Run code
"""
        )
        f.close()


def add_common_args(parser):
    parser.add_argument('script_name', type=str,
                        help='(str) name of .sh runscript file to produce (include ".sh")')
    parser.add_argument('runfile', type=str,
                        help='(str) file for cluster to execute (in current directory; include ".py")')
    parser.add_argument('n', type=int,
                        help='(int) n value')
    parser.add_argument('--env', type=str, default='pong',
                        help='(str) Name of Atari game. Default: pong')
    parser.add_argument('--timesteps', type=int, default=3_000_000,
                        help='(int) Training duration. Default: 3_000_000')
    parser.add_argument('--seed', type=int, default=0,
                        help='(int) Seed for random number generation. Default: 0')
    parser.add_argument('--batchmode', action='store_true',
                        help='(flag) Activates batch training if present. Default: disabled')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    if args.batchmode:
        batch_flag = True
    else:
        batch_flag = False

    main(args.script_name, args.runfile, args.env, args.n, args.timesteps, args.seed, batch_flag)


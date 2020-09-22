from argparse import ArgumentParser
from subprocess import Popen, check_output
import concurrent.futures
from threading import Lock
import os
import sys
import time
from itertools import chain
import yaml
from io import StringIO


class Dispatcher:
    def __init__(self):
        self.n_gpus = self._gpu_count()
        self._gpu_set = set(range(self.n_gpus))
        self._lock = Lock()

    def run(self, cmd, path, go):
        with self._lock:
            exists = self._report(cmd, path)

        if go and not exists:
            with open(path, 'w') as f:
                p = Popen(cmd, stdout=f, stderr=f)
                p.wait()

    def run_with_gpu(self, cmd, path, go, delay=0.0):
        with self._lock:
            exists = self._report(cmd, path)

            if go and not exists:
                with open(path, 'w') as f:
                    gpu_id = self._gpu_set.pop()
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    p = Popen(cmd, stdout=f, stderr=f)

        if go and not exists:
            p.wait()
            time.sleep(delay)  # Optional delay for GPU clean up

        with self._lock:
            self._gpu_set.add(gpu_id)

    def _report(self, cmd, path):
        print(' '.join(cmd))
        print('> ' + path)
        exists = os.path.exists(path)
        if exists:
            print('(file already exists)')
        print(flush=True)
        return exists

    def _gpu_count(self):
        try:
            return str(check_output(['nvidia-smi', '-L'])).count('UUID')
        except FileNotFoundError:
            return 0


def make_cmd(job, call_script):
    assert os.path.exists(call_script)
    cmd = ['python', call_script]
    for name, value in job:
        cmd += ['--' + name, value]
    return cmd


def make_filename(job):
    filename = []
    for name, value in job:
        s = '-'.join([name, value])
        filename += [s.replace('_', '')]
    return '_'.join(filename)


def submit_all(jobs, call_script, output_dir, go, auto_gpu):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cmd_list = [make_cmd(j, call_script) for j in jobs]
    path_list = [os.path.join(output_dir, make_filename(j)) for j in jobs]

    dispatcher = Dispatcher()
    if auto_gpu <= 0.0:
        for cmd, path in zip(cmd_list, path_list):
            dispatcher.run(cmd, path, go)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=dispatcher.n_gpus) as executor:
            for cmd, path in zip(cmd_list, path_list):
                executor.submit(dispatcher.run_with_gpu, cmd, path, go, auto_gpu)

    if not go:
        existing_paths = [p for p in path_list if os.path.exists(p)]

        print(f'Number of jobs:   {len(path_list):>5} total')
        print(f'                - {len(existing_paths):>5} exist')
        print(f'                = {len(path_list) - len(existing_paths):>5} to-do')
        print()
        print('*** This was just a test! No jobs were actually dispatched.')
        print('*** If the output looks correct, re-run with the "--go" argument.')
        print(flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--go', action='store_true')
    parser.add_argument('--auto-gpu', type=float, default=0.0,
                        help="(float) If greater than 0, feed all GPUs with specified delay (seconds).")
    args = parser.parse_args()

    def to_string(x):
        if isinstance(x, float):
            x = np.format_float_positional(x, trim='0', unique=True, precision=None)
        return str(x)

    def cartesian_product(manifest_yaml):
        # Load experiments
        print(f"Generating experiments from this yaml:\n---{manifest_yaml}...\n")
        experiments = yaml.safe_load(manifest_yaml).values()

        # Define a helper function (in case we have multiple experiments)
        def _cartesian_product(experiment):
            jobs = [[]]
            for name, values in experiment.items():
                if not isinstance(values, list):
                    # If there is only one value, wrap it in a list.
                    values = [values]
                # Generate the Cartesian product of the experiments with these new values.
                for _ in range(len(jobs)):
                    j = jobs.pop(0)
                    for v in values:
                        jobs.append(j + [(name, to_string(v))])
            return jobs

        # Call helper on each experiment
        return list(chain(*[_cartesian_product(e) for e in experiments]))

    def csv(values):
        return '[' + ','.join(to_string(v) for v in values) + ']'

    manifest_yaml = f"""
manifest:
    env:
      # classic 6 games:
      - breakout
      - beam_rider
      - pong
      - qbert
      - seaquest
      - space_invaders
      # worst games for proportional PER:
      - star_gunner
      - robotank
      # best games for proportional PER:
      - atlantis
      - gopher
    timesteps: 10_000_000
    rmem_type: [StratifiedReplayMemory, ReplayMemory]
    seed: {csv(range(5))}
"""
    jobs = cartesian_product(manifest_yaml)
    submit_all(jobs, 'dqn.py', args.output_dir, args.go, args.auto_gpu)


if __name__ == '__main__':
    main()

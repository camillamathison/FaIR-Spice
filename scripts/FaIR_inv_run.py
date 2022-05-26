import argparse
from itertools import islice
import multiprocessing
import os
import json
import time

import numpy as np

from fair.inverse import inverse_fair_scm

def numpy_json(json_obj:str, **kwargs):
    """Load an object from a JSON string and convert any  arrays to numpy arrays."""
    data = json.loads(json_obj, **kwargs)
    for k, v in data.items():
        # fair wants numpy arrays not python lists, so convert them
        if isinstance(v, list):
            data[k] = np.asarray(v)
    return data

def load_config_list(filename: str, start, end=None):
    """A python generator that reads one line of the config file at
    a time and converts to a python dictionary."""
    with open(filename, 'r') as fh:
        for line in islice(fh, start, end):
            yield numpy_json(line)

def run_fair(config):
    return inverse_fair_scm(**config)


def run(args):
    if os.path.exists(args.output):
        print("Loading previously saved data")
        output = np.load(args.output)
        total_runs = output.shape[0]
        # find the first output row where all values are 0
        start_point = np.argmin(np.all(output, axis=(1,2)))
        if start_point == 0:
            print("Nothing to do, exiting")
            return
    else:
        # count how many rows are in the input file. that will be the size
        # of the output file
        total_runs = sum(1 for _ in open(args.input))
        print("No output found. Creating new output file")
        output = np.zeros((total_runs, 3, 141))
        start_point = 0

    # create an iterator over the configs that only contains the run numbers we're computing
    configs = load_config_list(args.input, start_point)

    num_runs = total_runs - start_point

    print(f"Running {num_runs} runs on {args.num_cores} cores.")
    start = time.perf_counter()
    last_written = start_point
    try:
        with multiprocessing.Pool(args.num_cores) as pool:
            for i, result in enumerate(pool.imap(run_fair, configs)):
                cursor = start_point + i
                output[cursor, ...] = np.asarray(result)
                # after we've updated the output array update the
                # last_written value so if the process get's killed
                last_written = cursor
                if i % 1000 == 1:
                    now = time.perf_counter()
                    dt = now - start
                    speed = i / dt
                    remaining = num_runs - cursor
                    print(f"{cursor}/{num_runs} | {speed:.0f}it/sec | Est. {remaining/speed/60:.0f}min remaining")

    except KeyboardInterrupt:
        # this will also catch a 'SIGINT' signal.
        # with this SBATCH command in the submission script:
        # #SBATCH --signal=B:2@60
        # the job will be interrupted 60seconds before wallclock time runs
        # out, allowing time for the data to be saved
        print(f"Interrupted! Saving all output up to iteration {last_written}")

    finally:
        # blank out any partially written data that was in progress
        # when the KeyboardInterrupt occurred
        output[last_written+1:, ...] = 0

        stop = time.perf_counter()
        time_taken = stop - start
        num_run = last_written - start_point
        print(f"Ran {num_run} run in {time_taken:.0f}s | Averaged {num_run/time_taken:.2f}it/sec.")

        np.save(args.output, output)



if __name__ == '__main__':
    # work out how many cpu cores we have available
    N_CORES = int(os.environ.get('SLURM_NTASKS', '0')) or multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the list of configs')
    parser.add_argument('output', help='the filename storing the output')
    parser.add_argument('--num-cores', type=int, default=N_CORES)
    args = parser.parse_args()

    run(args)
#!/bin/bash -l

#SBATCH --get-user-env=L
#SBATCH --qos=normal
#SBATCH --time=200
#SBATCH --mem=32G
#SBATCH --ntasks=40
#SBATCH --output=/data/users/%u/fair_ensemble_%j.out
#SBATCH --signal=B:2@60

# To submit this job at the command line you need to write on the command line $> sbatch submit_job.sh
# I ran this once to make the config files and once to do the runs with co2 affecting temperature.
# Parallel to this I submitted the bgc_submit_job.sh to do the bgc runs where the CO2 is not allowed to affect temperature.

# $> sacct allows progress of job on spice to be monitored
# $> scancel allows you to cancel a job if you need to

# Only one exec command will run at a time they won't run consecutively from one submit.
# To minimise how long it takes to run it is recommended to copy this script and rename it and resubmit it for each set of configs.

set -euo pipefail
conda activate fair-spice
#exec python FaIR_inv_make_configs.py # to make the config files
exec python scripts/run_ensemble.py --samples=100000 ensemble_full.nc
#exec python FaIR_inv_run.py configsdir/bgc_configs.json data_output/bgc_inverse_results.np


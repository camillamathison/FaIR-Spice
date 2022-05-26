#!/bin/bash -l

#SBATCH --get-user-env=L
#SBATCH --qos=normal
#SBATCH --time=200
#SBATCH --mem=32G
#SBATCH --ntasks=26
#SBATCH --output=/data/users/hadcam/qjob.out
#SBATCH --signal=B:2@60

# To submit this job at the command line you need to write on the command line $> sbatch submit_job.sh
# I ran this once to make the config files and once to do the runs with co2 affecting temperature. 
# Parallel to this I submitted the bgc_submit_job.sh to do the bgc runs where the CO2 is not allowed to affect temperature.  

# $> sacct allows progress of job on spice to be monitored 
# $> scancel allows you to cancel a job if you need to

# Only one exec command will run at a time they won't run consecutively from one submit. 
# To minimise how long it takes to run it is recommended to copy this script and rename it and resubmit it for each set of configs.

set -euo pipefail
conda activate climate-outcomes
#exec python FaIR_inv_make_configs.py # to make the config files
exec python FaIR_inv_run.py configsdir/std_configs.json ../data_output/std_inverse_results.npy 
#exec python FaIR_inv_run.py configsdir/bgc_configs.json data_output/bgc_inverse_results.np


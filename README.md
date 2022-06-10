# FaIR-Spice
A repository to run big ensembles of FaIR using Spice and save output and configs. 
In this repository you will find example notebooks with manageable examples of running FaIR and examples to 
run much bigger ensembles of runs on Spice using scripts.

## Conda 

The conda environment you need to run the code is in environment.yml

## The jupyter notebooks in /notebooks provides several example notebooks:

run_ensemble.ipynb - provides three options for running FaIR in the inverse set up from concentrations to diagnose emissions and provide a temperature.
                   - uses xarray to enable the addition of metadata so you can easily plot the data that comes out as shown in the xarray_analysis.ipynb notebook 
xarray_analysis.ipynb - notebook to plot the output
simple_forward.ipynb - a really simple example for running FaIR from emissions to give a concentration, forcing and temperature.  
simple_inverse.ipynb - a really simple example for running FaIR from concentrations to give emissions, forcing and temperature.  


## The code in /scripts provides a way of running much bigger ensembles directly on Spice
 
run_ensemble.py:   The configs for running FaIR are created in this file. 
                   You can change this file to specify the run. 
                   I recommend changing the name of this file to reflect the run you have set up e.g.,  run_ensemble_std.py for a standard run where CO2 affects climate and run_ensemble_bgc.py for a run where the CO2 is not allowed to affect climate:

submit_job.sh:      This script can be submitted to spice using the sbatch command. 
                    Edit this to do a different size ensemble and to make sure you don't overwrite your data. 




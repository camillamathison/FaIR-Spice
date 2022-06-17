# FaIR-Spice
A repository to run big ensembles of FaIR using Spice and save output and configs. 
In this repository you will find example notebooks with manageable examples of running FaIR and examples to 
run much bigger ensembles of runs on Spice using scripts.

## Conda 

The conda environment you need to run the code is in environment.yml

Recommended to use python 3.7+ with anaconda.

1. Navigate to your local working directory, then clone the repository
```bash
git clone https://github.com/camillamathison/FaIR-Spice.git
cd Fair-Spice
```

2. Create environment:
```bash
conda env create -f environment.yml
```

3. To run ...
```bash
conda activate fair-spice
```

## The jupyter notebooks in /notebooks provides several example notebooks:

- run_ensemble.ipynb 
    - provides three options for running FaIR in the inverse set up from concentrations to diagnose emissions and provide a temperature.
    - uses xarray to enable the addition of metadata so you can easily plot the data that comes out as shown in the xarray_analysis.ipynb notebook 
- xarray_analysis.ipynb - notebook to plot the output
- xarray_analysis_alphabetagamma_ensemble.ipynb - notebook to plot the output from a bgc and standard run to calculate alpha, beta and gamma
- simple_forward.ipynb - a really simple example for running FaIR from emissions to give a concentration, forcing and temperature.  
- simple_inverse.ipynb - a really simple example for running FaIR from concentrations to give emissions, forcing and temperature.  
- run-fair-ar6-ssp245.ipynb 
    - A more complicated example using the constrained AR6 set up of FaIR 1.6 to run two ssp scenarios (SSP2-45 and SSP3-60)
    - These two scenarios are relevant to work done already on the Methane pledges .  



## The code in /scripts provides a way of running much bigger ensembles directly on Spice
 
**run_ensemble.py:**
The configs for running FaIR are created in this file. you can run this using this command:

e.g. running a sample size of 1000 you would use this command:
```bash
python scripts/run_ensemble.py --samples=1000 "fair_ensemble.nc"
```

You can change this file to specify the details of the run. 
I recommend changing the name of this file to reflect the run you have set up                    
                   
e.g., An example of the commands you would use to run a 10000 member ensemble for a standard run where CO2 affects climate and 
```bash
python scripts/run_ensemble_std.py --samples=10000 "fair_ensemble_std.nc"
```
and perhaps to run an equivalent size ensemble where the CO2 is not allowed to affect climate: e.g.
```bash
python scripts/run_ensemble_bgc.py --samples=10000 "fair_ensemble_bgc.nc"
```
This would create two lots of output for similar runs but make sure one would not overwrite the other.
I have used this to run a 1000000 runs on Spice which can be submitted using the submit_job.sh script below.
                   
**submit_job.sh:**
The run_ensemble.py script can be submitted to spice using the sbatch command. 
Edit this to do a different size ensemble and to make sure you don't overwrite your data.
```bash
sbatch submit_job.sh
```



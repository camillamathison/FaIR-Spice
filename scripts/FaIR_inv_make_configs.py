import json
import numpy as np
import pandas as pd
import fair
from climateforcing.utils import mkdir_p
from tqdm import tqdm
from multiprocessing import Pool
import scipy.stats as st
from fair.forcing.ghg import co2_log

fair.__version__   # needs to be 1.4+

'''
 Define the 1% CO2 forcing time series. Because of the limitation of inverse fair,
 I don't think we can select Etminan or Meinshausen forcing and are limited to the logarithmic relationship.
 Given recent work on forcing uncertainties in GCMs, this is probably not a huge limitation.
 '''

nt = 141
Cpi = 284.32
C = Cpi*1.01**np.arange(nt)

# Load in the inputs we need
with open('../../COMMON_AR6_INFILES/random_seeds.json', 'r') as filehandle:
    SEEDS = json.load(filehandle)

SAMPLES = 100000  #100000
F2XCO2_MEAN = 4.00
F2XCO2_NINETY = 0.48
NINETY_TO_ONESIGMA = st.norm.ppf(0.95)

# Make the locations for the output data
name_samplesize = 'samples_'+str(SAMPLES)+'_full'
mkdir_p('../data_output/fair-samples/'+name_samplesize+'/')
mkdir_p('../data_output/fair-samples/'+name_samplesize+'/alpha_beta_gamma/')
mkdir_p('../data_inputs/fair-samples/'+name_samplesize+'/')

with open("../../COMMON_AR6_INFILES/tunings/cmip6_twolayer_tuning_params.json", "r") as read_file:
    params = json.load(read_file)
cmip6_models = list(params['q4x']['model_data']['EBM-epsilon'].keys())

NMODELS = len(cmip6_models)

geoff_data = np.zeros((NMODELS, 6))
for im, model in enumerate(cmip6_models):
    geoff_data[im,0] = params['q4x']['model_data']['EBM-epsilon'][model]
    geoff_data[im,1] = params['lamg']['model_data']['EBM-epsilon'][model]
    geoff_data[im,2] = params['cmix']['model_data']['EBM-epsilon'][model]
    geoff_data[im,3] = params['cdeep']['model_data']['EBM-epsilon'][model]
    geoff_data[im,4] = params['gamma_2l']['model_data']['EBM-epsilon'][model]
    geoff_data[im,5] = params['eff']['model_data']['EBM-epsilon'][model]

geoff_df = pd.DataFrame(geoff_data, columns=['q4x','lamg','cmix','cdeep','gamma_2l','eff'], index=cmip6_models)

print("Read model input data")

kde = st.gaussian_kde(geoff_df.T)
geoff_sample = kde.resample(size=int(SAMPLES*1.25), seed = SEEDS[15])

# remove unphysical combinations
geoff_sample[:,geoff_sample[0,:] <= 0] = np.nan
#geoff_sample[:,geoff_sample[1,:] >= -0.6] = np.nan
geoff_sample[1, :] = st.truncnorm.rvs(-2, 2, loc=-4/3, scale=0.5, size=int(SAMPLES*1.25), random_state=SEEDS[16])
geoff_sample[:,geoff_sample[2,:] <= 0] = np.nan
geoff_sample[:,geoff_sample[3,:] <= 0] = np.nan
geoff_sample[:,geoff_sample[4,:] <= 0] = np.nan
geoff_sample[:,geoff_sample[5,:] <= 0] = np.nan

mask = np.all(np.isnan(geoff_sample), axis=0)
geoff_sample = geoff_sample[:,~mask][:,:SAMPLES]
geoff_sample_df=pd.DataFrame(data=geoff_sample.T, columns=['q4x','lamg','cmix','cdeep','gamma_2l','eff'])
geoff_sample_df.to_csv('../data_inputs/'+name_samplesize+'geoff_sample.csv')

f2x = st.norm.rvs(loc=F2XCO2_MEAN, scale=F2XCO2_NINETY/NINETY_TO_ONESIGMA, size=SAMPLES, random_state=SEEDS[73])

ecs = -f2x/geoff_sample[1,:]
tcr = f2x/(-geoff_sample[1,:] + geoff_sample[4,:]*geoff_sample[5,:])


np.save('../data_inputs/fair-samples/'+name_samplesize+'/C_unconstrained.npy', C)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/f2x_unconstrained.npy', f2x)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/ecs_unconstrained.npy', ecs)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/tcr_unconstrained.npy', tcr)

#Important change relative to AR6:
# the carbon cycle feedback on airborne fraction can admit small negative values. In the constrained AR6,
# this only ever occurs with a high rT (temperature feedback).
# therefore, rC_fs below is set to have a lower bound of zero and an upper bound of 0.0482.

r0_fs = st.uniform.rvs(loc=27.7, scale=41.3-27.7, random_state=SEEDS[10], size=SAMPLES)
rC_fs = st.uniform.rvs(loc=0, scale=0.0482, random_state=SEEDS[11], size=SAMPLES)
rT_fs = st.uniform.rvs(loc=-0.0847, scale=4.52+0.0847, random_state=SEEDS[12], size=SAMPLES)
pre_ind_co2 = st.norm.rvs(loc=277.147, scale=2.9, random_state=SEEDS[13], size=SAMPLES)

np.save('../data_inputs/fair-samples/'+name_samplesize+'/r0_unconstrained.npy', r0_fs)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/rC_unconstrained.npy', rC_fs)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/rT_unconstrained.npy', rT_fs)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/pre_ind_co2_unconstrained.npy', pre_ind_co2)

print("Wrote fair-samples:")
print(rT_fs.shape, r0_fs.shape, rC_fs.shape)
print(rT_fs[1:5], r0_fs[1:5], rT_fs[1:5])

tcrecs_un = np.zeros((SAMPLES, 2))

for i in range(SAMPLES):

    #r0_un[i] = r0_fs[i]                                      #    r0[i] = config[i]['r0']
    #rc_un[i] = rC_fs[i]                                      #    rc[i] = config[i]['rc']
    #rt_un[i] = rT_fs[i]                                      #    rt[i] = config[i]['rt']
    #f2x_un[i] = f2x[i]                                       #    f2x[i] = config[i]['F2x']

    gamma_un = geoff_sample_df.loc[i, 'gamma_2l']      #    gamma = config[i]['ocean_heat_exchange']
    feedback_un = -geoff_sample_df.loc[i, 'lamg']      #    feedback = config[i]['lambda_global']
    eff_un = geoff_sample_df.loc[i, 'eff']             #    eff = config[i]['deep_ocean_efficacy']

    ecs_un = f2x[i]/feedback_un                       #    ecs = f2x[i]/feedback
    tcr_un = f2x[i]/(feedback_un + (gamma_un * eff_un)) #    tcr = f2x[i]/(feedback + (gamma * eff))
    tcrecs_un[i] = [tcr_un, ecs_un]                         #    tcrecs[i] = [tcr, ecs]



#Run the first simulations in which the carbon is allowed to affect temperature as normal.

result = {}
std_config = []

for i in range(SAMPLES):
    cfg = {}
    cfg['C'] = C
    cfg['tcrecs'] = tcrecs_un[i,:]
    cfg['r0'] = r0_fs[i]   #r0_un[i],
    cfg['rc'] = rC_fs[i]   #rc_un[i],
    cfg['rt'] = rT_fs[i]   #rt_un[i],
    cfg['F2x'] = f2x[i]
    cfg['other_rf'] = 0.

    std_config.append(cfg)

# use pandas to write the config file as a list of json dicts
# this means we can load it in again one row at a time so won't have to
# hold the whole set of config dicts in memory all in one go
config_df = pd.DataFrame(std_config)
config_df.to_json('configsdir/std_configs.json', orient='records', lines=True)

print(f"Wrote {len(std_config)} configurations to data_inputs/std_configs.json")

bgc_config = []

for i in range(SAMPLES):
    cfg = {}
    cfg['C'] = C
    cfg['tcrecs'] = tcrecs_un[i,:]
    cfg['r0'] = r0_fs[i]   #r0_un[i],
    cfg['rc'] = rC_fs[i]   #rc_un[i],
    cfg['rt'] = rT_fs[i]   #rt_un[i],
    cfg['F2x'] = f2x[i]
    cfg['other_rf'] = -co2_log(C, Cpi, F2x=f2x[i])
    
    bgc_config.append(cfg)

config_df = pd.DataFrame(bgc_config)
config_df.to_json('configsdir/bgc_configs.json', orient='records', lines=True)

print(f"Wrote {len(bgc_config)} configurations to configsdir/bgc_configs.json")

# new config holding ecs constant

ecs_config = []

for i in range(SAMPLES):

    #r0_un[i] = r0_fs[i]                                      #    r0[i] = config[i]['r0']
    #rc_un[i] = rC_fs[i]                                      #    rc[i] = config[i]['rc']
    #rt_un[i] = rT_fs[i]                                      #    rt[i] = config[i]['rt']
    #f2x_un[i] = f2x[i]                                       #    f2x[i] = config[i]['F2x']

    gamma_un = geoff_sample_df.loc[i, 'gamma_2l']      #    gamma = config[i]['ocean_heat_exchange']
    feedback_un = 1.25                                 #    feedback = config[i]['lambda_global']
    eff_un = geoff_sample_df.loc[i, 'eff']             #    eff = config[i]['deep_ocean_efficacy']

    ecs_un = f2x[i]/feedback_un                       #    ecs = f2x[i]/feedback
    tcr_un = f2x[i]/(feedback_un + (gamma_un * eff_un)) #    tcr = f2x[i]/(feedback + (gamma * eff))
    tcrecs_un[i] = [tcr_un, ecs_un]                         #    tcrecs[i] = [tcr, ecs]

np.save('../data_inputs/fair-samples/'+name_samplesize+'/feedfix_ecs_unconstrained.npy', ecs_un)
np.save('../data_inputs/fair-samples/'+name_samplesize+'/feedfix_tcr_unconstrained.npy', tcr_un)

for i in range(SAMPLES):
    cfg = {}
    cfg['C'] = C
    cfg['tcrecs'] = tcrecs_un[i,:]
    cfg['r0'] = r0_fs[i]   #r0_un[i],
    cfg['rc'] = rC_fs[i]   #rc_un[i],
    cfg['rt'] = rT_fs[i]   #rt_un[i],
    cfg['F2x'] = f2x[i]
    cfg['other_rf'] = 0.

    ecs_config.append(cfg)

# use pandas to write the config file as a list of json dicts
# this means we can load it in again one row at a time so won't have to
# hold the whole set of config dicts in memory all in one go
config_df = pd.DataFrame(ecs_config)
config_df.to_json('configsdir/ecs_configs.json', orient='records', lines=True)

print(f"Wrote {len(ecs_config)} configurations to configsdir/ecs_configs.json")
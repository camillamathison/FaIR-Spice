

from fair_spice.ensemble import make_member, stream_ensemble
import fair_spice.config as config
from fair_spice.fair.models import inverse_fair

def run_model(params):
    n, config = params
    run = inverse_fair(config)
    member = make_member(run, n, invariant=["C"])
    return member

def main():
    import argparse
    import multiprocessing
    import os
    import time
    import sys

    import numpy as np
    import pandas as pd
    import scipy.stats as st
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile")
    # parser.add_argument("--partition", "-p")
    # parser.add_argument("--partition-size", "-s")
    parser.add_argument("--num-cores", "-n")
    parser.add_argument("--samples", type=int, default=1000000)

    args = parser.parse_args()

    outfile = args.outfile

    N_PROCS = args.num_cores or int(os.environ.get("SLURM_NTASKS", "0")) or os.cpu_count()
    SEEDS = config.RANDOM_SEEDS
    SAMPLES = args.samples
    F2XCO2_MEAN = 4.00
    F2XCO2_NINETY = 0.48
    NINETY_TO_ONESIGMA = st.norm.ppf(0.95)

    nt = 141
    Cpi = 284.32
    C = Cpi * 1.01 ** np.arange(nt)

    r0_fs = st.uniform.rvs(
        loc=27.7, scale=41.3 - 27.7, random_state=SEEDS[10], size=SAMPLES
    )
    rC_fs = st.uniform.rvs(loc=0, scale=0.0482, random_state=SEEDS[11], size=SAMPLES)
    rT_fs = st.uniform.rvs(
        loc=-0.0847, scale=4.52 + 0.0847, random_state=SEEDS[12], size=SAMPLES
    )
    pre_ind_co2 = st.norm.rvs(loc=277.147, scale=2.9, random_state=SEEDS[13], size=SAMPLES)

    tunings_file = config.ROOT_DIR / "data_input/tunings/cmip6_twolayer_tuning_params.json"
    column_order = ["q4x", "lamg", "cmix", "cdeep", "gamma_2l", "eff"]
    tunings = config.load_tunings(tunings_file).drop(columns=["t4x"])[column_order]

    print("First 5 model parameters:")
    print(tunings.head())
    print()


    # Create a 6D Gaussian kernel based on the data from the model tunings
    # and resample for the number of samples we want to cover
    kde = st.gaussian_kde(tunings.T)
    samples_raw = kde.resample(size=int(SAMPLES * 1.25), seed=SEEDS[15])

    samples = pd.DataFrame(samples_raw.T, columns=tunings.columns)
    # remove unphysical combinations
    samples[samples <= 0] = np.nan
    # lamg is always < 0. But this truncnorm approx seems to overwrite
    # sample distribution that came from the kde? TODO: check
    samples.lamg = st.truncnorm.rvs(
        -2, 2, loc=-4 / 3, scale=0.5, size=int(SAMPLES * 1.25), random_state=SEEDS[16]
    )

    samples = samples.dropna(how="any")[:SAMPLES]
    print("First 5 param samples")
    print(samples.head())
    print()

    # create TCR and ECS random samples from the model distributions
    f2x = st.norm.rvs(
        loc=F2XCO2_MEAN,
        scale=F2XCO2_NINETY / NINETY_TO_ONESIGMA,
        size=SAMPLES,
        random_state=SEEDS[73],
    )

    ecs = -f2x / samples.lamg
    tcr = f2x / (-samples.lamg + samples.gamma_2l * samples.eff)

    # create all the configs as a dataframe
    config_table = pd.DataFrame(
        dict(r0=r0_fs, rc=rC_fs, rt=rT_fs, F2x=f2x, tcr=tcr, ecs=ecs, other_rf=0.0)
    )
    print("First 5 configs:")
    print(config_table.head())


    configs = ({**x._asdict(), "C": C} for x in config_table.itertuples(index=False))

    print(f"Running {SAMPLES} ensemble members on {N_PROCS} cores:")
    configs = tqdm(configs, total=SAMPLES, ncols=80, file=sys.stdout)
    with multiprocessing.Pool(N_PROCS, maxtasksperchild=1000) as pool:
        batch = pool.imap(run_model, enumerate(configs))
        stream_ensemble(outfile, batch)

    print(f"Done. Output saved to {outfile}")

if __name__ == '__main__':
    main()

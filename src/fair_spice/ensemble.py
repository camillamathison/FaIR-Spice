from functools import partial
import multiprocessing

import netCDF4
import numpy as np
from tqdm import tqdm


def write_data(group, run_num, run_data, config):
    for var, data in run_data.items():
        group.variables[var][run_num, :] = data
    for param, value in config.items():
        group.variables[param][run_num] = value


def _echo_model(config, *, model, forcing):
    output = model(**forcing, **config)
    return output, config

def setup_dataset(dset, run0, cfg0, forcing):
    """Using a single run and configuration as a template,
    Setup a netcdf Dataset to store ensemble members."""

    # find the length of the timeseries from the results returned from the first
    # run of the model
    year_len = max(len(x) for x in run0.values())

    dset.createDimension("member", None)   # unlimited length
    dset.createDimension("year", year_len)

    dset.createVariable("member", int, ("member",))
    years = dset.createVariable("year", int, ("year",))
    years[:] = np.arange(len(years))+1

    for var, data in run0.items():
        dset.createVariable(var, data.dtype, ('member', 'year'))

    # Save the ensemble input parameter values as netcdf member variables
    for param, value in cfg0.items():
        dset.createVariable(param, type(value), ('member',))

    # Save the forcing input values as netcdf year variables
    for var, _data in forcing.items():
        data = np.asarray(_data)
        if data.shape == years.shape:
            forcing = dset.createVariable(var, data.dtype, ("year", ))
            forcing[:] = data
        else:
            forcing = dset.createVariable(var, type(_data), ())
            forcing[:] = _data


def run_ensemble(model, configs, forcing, outfile='ensemble.nc', nproc=4, tqdm=tqdm):
    """Run an ensemble of model configurations and save to a netcdf file.

    The netCDF Dataset will have two dimensions:
      1. member: the ensemble member
      2. year: the model output timeseries

    Anything passed into the model as a `config` will be stored on the
    member dimension. Currently only supports scalar config values
    Anything passed into the model as a `forcing` will be stored on the
    year dimension if it has time-dimensionality.

    Model should return a dict of timeseries keyed with their variable name,
    e.g. {'temperature': [0.1, 0.11, ...]}. These are saved as separate
    variables in the output.

    Arguments
    ---------
        model: A callable that takes a config object as a single argument
               and returns a dict of timeseries values to be saved
               {'temperature': [0.1, ...]}
        configs: An iterable of config objects containing keyword arguments to
                 be passed to the model that define the input to each ensemble
                 member.
        forcing: A dict of static forcing variables that do not vary across the
                 ensemble members. For example, when running the inverse fair
                 scm, this could be the concentration timeseries {'C': [284.32,
                 ...]}. Any duplicate key in both config and forcing dicts will
                 use the value from config and override the static forcing.


    Returns
    -------
        None
    """
    iter_cfg = iter(configs)

    # we want to send the model function, including static forcings,
    # to multiprocessing nodes, so it needs to be
    # pickleable. `partial` is compatible with pickle.
    _model = partial(_echo_model, model=model, forcing=forcing)

    # TODO this could be improved to support non-scalar configs too
    with netCDF4.Dataset(outfile, "w", format="NETCDF4") as dset:
        # run the model once to determine the output shape and size
        # of the output data and config variables and use that to setup
        # the netcdf Dataset
        run0, cfg0 = _model(next(iter_cfg))
        setup_dataset(dset, run0, cfg0, forcing)

        # write the first set of values from the first iteration
        write_data(dset, 0, run0, cfg0)

        # using a pool of workers to run the ensemble
        # data storage is synchronous - only the main process writes to the
        # netcdf file
        with multiprocessing.Pool(nproc) as pool:
            for i, (res, cfg) in enumerate(pool.imap(_model, iter_cfg)):
                try:
                    run_num = i+1
                    write_data(dset, run_num, res, cfg)
                except KeyboardInterrupt:
                    print("interrupting! Saving data...")
                    dset.sync()









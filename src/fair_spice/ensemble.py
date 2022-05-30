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

def run_ensemble(model, configs, forcing, outfile='ensemble.nc', year0=1850, nproc=4):
    """

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
        The path to a netcdf file containing the ensemble data
    """

    # zip together the per-run config and the static forcing
    iter_cfg = iter(configs) # ({**cfg, **forcing} for cfg in iter(configs))

    # run the model once to determine the output shape and size
    # of the output data and config variables

    _model = partial(_echo_model, model=model, forcing=forcing)

    run0, cfg0 = _model(next(iter_cfg))

    # find the length of the timeseries from the results returned from the first
    # run of the model
    year_len = max(len(x) for x in run0.values())

    # create a netcdf file to write the data to
    # this dataset will have two dimensions:
    #   1. member: the ensemble member
    #   2. year: the model output timeseries
    # Anything passed into the model as a `config` will be stored on the
    # member dimension. currently only supports scalar config values
    # Anything passed into the model as a `forcing` will be stored on the
    # year dimension if it has time-dimensionality.

    # TODO this could be improved to support non-scalar configs too
    dset = netCDF4.Dataset(outfile, "w", format="NETCDF4")
    dset.createDimension("member", None)
    dset.createDimension("year", year_len)

    years = dset.createVariable("year", int, ("year",))
    years[:] = np.arange(year_len) + year0

    for var, data in run0.items():
        dset.createVariable(var, data.dtype, ('member', 'year'))

    # also store the input parameter values as netcdf member variables
    for param, value in cfg0.items():
        dset.createVariable(param, type(value), ('member',))

    # store the forcing values as netcdf year variables
    for var, _data in forcing.items():
        data = np.asarray(_data)
        if data.shape == years.shape:
            forcing = dset.createVariable(var, data.dtype, ("year", ))
            forcing[:] = data
        else:
            forcing = dset.createVariable(var, type(_data), ())
            forcing[:] = _data

    # write the first set of values from the first iteration
    write_data(dset, 0, run0, cfg0)

    with multiprocessing.Pool(nproc) as pool:
        for i, (res, cfg) in enumerate(tqdm(pool.imap(_model, iter_cfg))):
            try:
                run_num = i+1
                write_data(dset, run_num, res, cfg)
            except KeyboardInterrupt:
                print("interrupting! Saving data...")
                dset.sync()

    dset.close()
    return outfile








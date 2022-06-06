from typing import Optional
import numpy as np
import xarray as xr

from fair.inverse import inverse_fair_scm


def inverse_fair(config: dict, forcing: Optional[dict] = None) -> xr.Dataset:
    """Run the inverse_fair_scm model and return an xarray Dataset.

    Arguments:
        config: A dict of scalar input parameters. These are passed to
                inverse_fair_scm as keyword arguments. They will be coordinate
                values in the returned Dataset
        forcing: An optional dict of static input parameters. These are passed
                 to inverse_fair_scm as keyword arguments.

        If a forcing is given an additional empty 'ensemble_member' dimension is
        added to the cube. All config entries are member-dependent dimension values, all `forcing` entries are independent of the member dimension. This simplifies running ensembles of model runs where some config options are varied across the ensemble and others remain constant.


        Currently does not support the additional return types for the
        'Geoffrey' temperature profile or restarts.

    Returns:
       xarray.Dataset of the model run
    """
    if forcing is None:
        forcing = {}

    cfg = {**forcing, **config}
    # we want to record tcr and ecs separately, but the inverse_fair_scm model
    # expects them to be passed in together as an array. So we'll pop them out
    # and create the `tcrecs` object before running the function
    if "tcr" in cfg:
        tcr = cfg.pop("tcr")
        ecs = cfg.pop("ecs")
        cfg["tcrecs"] = np.array([tcr, ecs])

    E, F, T = inverse_fair_scm(**cfg)

    dset = xr.Dataset(
        {
            "temperature": xr.Variable(("time",), T, {"units": "K"}),
            "forcing": xr.Variable(("time",), F, {"units": "W/m2"}),
            "emissions": xr.Variable(("time",), E, {"units": "GtC"}),
        },
        coords={
            "time": xr.Variable(("time",), np.arange(len(E)), {"units": "year"}),
            **config,
            **forcing,
            "C": xr.Variable(("time",), cfg["C"], {"units": "ppmv[CO2]"}),
        },
    )

    # if the static forcing is applied separately to config, create
    # a new ensemble_member dimension and apply to all config entries
    if forcing:
        dset = dset.expand_dims("ensemble_member")
        for c in config:
            dset[c] = dset[c].expand_dims("ensemble_member")

    return dset


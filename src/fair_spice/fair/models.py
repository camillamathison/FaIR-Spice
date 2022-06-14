from typing import Any, Optional
import numpy as np
import xarray as xr

from fair.forward import fair_scm
from fair.inverse import inverse_fair_scm

from fair_spice.fair.constants import cmip_gases, forcing_order

_forward_meta = {
    "time": {"units": "year"},
    "temperature": {"units": "degC", "description": "Temperature anomaly"},
    "radiative_forcing": {
        "units": "W/m2",
        "description": "Other radiative forcing in W/m2",
    },
    "concentration": {
        "units": "ppm",
        "description": "Atmospheric concentration of greenhouse gases",
    },
}


def forward_fair(config: dict[str, Any]) -> xr.Dataset:
    """Run the fair_scm model and return an xarray Dataset.

    Arguments:
        config: A dict of scalar input parameters. These are passed to
                fair_scm as keyword arguments. They will be coordinate
                values in the returned Dataset

    Currently does not support:
     * diagnostics="AR6"

    Returns:
       xarray.Dataset of the fair_scm model run
    """
    C, F, T = fair_scm(**config)

    coords = {"time": xr.Variable(("time",), np.arange(len(T)))}

    # handle multi-gas case
    if len(C.shape) == 2:
        cdims = ("time", "gas")
        fdims = ("time", "forcing")
        coords["gas"] = xr.Variable(("gas",), cmip_gases)
        coords["forcing"] = xr.Variable(("forcing",), forcing_order)
    else:
        cdims = fdims = ("time",)

    data_vars = {
        "temperature": xr.Variable(("time",), T),
        "radiative_forcing": xr.Variable(fdims, F),
        "concentration": xr.Variable(cdims, C),
    }

    dset = xr.Dataset(data_vars, coords)
    _apply_meta(dset, _forward_meta)

    return dset


_inverse_meta = {
    "time": {"units": "year"},
    "emissions": {"units": "GtC", "long_name": "Diagnosed CO2 emissions"},
    "forcing": {"units": "W/m2", "long_name": "Total radiative forcing"},
    "temperature": {
        "units": "degC",
        "long_name": "Temperature anomaly since pre-industrial",
    },
    "tcr": {"units": "degC", "long_name": "Transient Climate Response"},
    "ecs": {"units": "degC", "long_name": "Equilibrium Climate Response"},
    "C": {"units": "ppmv[CO2]"},
    "other_rf": {"units": "W/m2"},
    "F2x": {"units": "W/m2"},
    "rc": {"units": "yr/GtC"},
}


def inverse_fair(config: dict[str, Any]) -> xr.Dataset:
    """Run the inverse_fair_scm model and return an xarray Dataset.

    Arguments:
        config: A dict of scalar input parameters. These are passed to
                inverse_fair_scm as keyword arguments. They will be coordinate
                values in the returned Dataset

        Currently does not support the additional return types for the
        'Geoffrey' temperature profile or restarts.

        The only difference from the underlying fair_inverse_fcm is that
        `tcr` and `ecs` are passed as separate config parameters rather
        than single 2-d array.

    Returns:
       xarray.Dataset of the inverse model run
    """

    cfg = {**config}
    # we want to record tcr and ecs separately, but the inverse_fair_scm model
    # expects them to be passed in together as an array. So we'll pop them out
    # and create the `tcrecs` object before running the model
    if "tcr" in cfg:
        tcr = cfg.pop("tcr")
        ecs = cfg.pop("ecs")
        cfg["tcrecs"] = np.array([tcr, ecs])

    E, F, T = inverse_fair_scm(**cfg)
    data_vars = {
        "temperature": xr.Variable(("time",), T),
        "forcing": xr.Variable(("time",), F),
        "emissions": xr.Variable(("time",), E),
    }

    years = np.arange(len(E))
    coords = {"time": xr.Variable(("time",), years)}
    # turn the model input parameters into coordinate variables
    for param, value in config.items():
        if np.asarray(value).shape == years.shape:
            dims = ("time",)
        else:
            dims = ()
        coords[param] = xr.Variable(dims, value)

    dset = xr.Dataset(data_vars, coords)
    # add metadata to all the variables and coordinates
    _apply_meta(dset, _inverse_meta)

    return dset


def _apply_meta(dset: xr.Dataset, meta: dict[str, Any]):
    for name, var in dset.variables.items():
        var.attrs.update(meta.get(name, {}))

from typing import Any, Optional
import numpy as np
import xarray as xr

from fair.forward import fair_scm
from fair.inverse import inverse_fair_scm

from fair_spice.constants import cmip_gases, forcing_order


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

    coords = {"time": xr.Variable(("time",), np.arange(len(T)), {"units": "year"})}

    # handle multi-gas case
    if len(C.shape) == 2:
        cdims = ("time", "gas")
        fdims = ("time", "forcing")
        coords["gas"] = xr.Variable(("gas",), cmip_gases)
        coords["forcing"] = xr.Variable(("forcing",), forcing_order)
    else:
        cdims = fdims = ("time",)

    dset = xr.Dataset(
        {
            "temperature": xr.Variable(
                ("time",),
                T,
                {"units": "K", "description": "Temperature anomaly in Kelvin"},
            ),
            "radiative_forcing": xr.Variable(
                fdims,
                F,
                {"units": "W/m2", "description": "Other radiative forcing in W/m2"},
            ),
            "concentration": xr.Variable(
                cdims,
                C,
                {
                    "units": "ppm",
                    "description": "Atmospheric concentration of greenhouse gases",
                },
            ),
        },
        coords=coords,
    )

    return dset


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
            "C": xr.Variable(("time",), cfg["C"], {"units": "ppmv[CO2]"}),
        },
    )

    return dset

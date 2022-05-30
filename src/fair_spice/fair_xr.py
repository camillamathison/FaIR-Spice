"""
A thin wrapper around fair to return xarray objects instead of bare numpy arrays


"""
from functools import wraps

import fair
import xarray as xr

# @wraps(fair.forward.fair_scm)
# def fair_scm(*args, **kwargs):
#     res = fair.forward.fair_scm(*args, **kwargs)


@wraps(fair.inverse.inverse_fair_scm)
def inverse_fair_scm(*args, **kwargs):
    E, F, T = fair.inverse.inverse_fair_scm(*args, **kwargs)
    return xr.Dataset()
import numpy as np
import pandas as pd

cmip_gases = ["CO2", "CH4", "N2O", "CF4", "C2F6", "C6F14", "HFC23", "HFC32", "HFC43_10", "HFC125",
            "HFC134A", "HFC143A", "HFC227EA", "HFC245FA", "SF6", "CFC11", "CFC12", "CFC113",
            "CFC114", "CFC115", "CARB_TET", "MCF", "HCFC22", "HCFC141B", "HCFC142B",
            "HALON1211", "HALON1202", "HALON1301", "HALON2402", "CH3BR", "CH3CL"]

forcing_order = ["CO2", "CH4", "N2O", "other_ghg", "trop_ozone", "strat_ozone",
            "strat_H2O", "contrails", "aerosols", "black_C", "LUC", "volcanic",
            "solar"]

from fair.constants import radeff, lifetime, molwt

def _series_from_module(mod, constants):
    return [mod.__dict__.get(c, np.nan) for c in constants]

# Table 2 from Geosci. Model Dev., 11, 2273–2297, 2018
ghg = pd.DataFrame(dict(
    gas=cmip_gases,
    molwt=_series_from_module(molwt, cmip_gases),
    radeff=_series_from_module(radeff, cmip_gases),
    lifetime=_series_from_module(lifetime, cmip_gases),
    ))

# Table 3 from Geosci. Model Dev., 11, 2273–2297, 2018
forcings = pd.DataFrame(dict(
    forcing_agent=forcing_order
))
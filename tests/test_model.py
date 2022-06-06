import numpy as np

from fair_spice.models import forward_fair, inverse_fair

from fair.RCPs import rcp3pd

def test_forward_simple():
    emissions = np.zeros(250)
    emissions[125:] = 10.0
    other_rf = np.zeros(emissions.size)
    for x in range(0, emissions.size):
        other_rf[x] = 0.5 * np.sin(2 * np.pi * (x) / 14.0)

    run = forward_fair(dict(
        emissions=emissions,
        other_rf=other_rf,
        useMultigas=False
    ))

    assert run.concentration.shape == (250,)
    assert run.concentration.units == 'ppm'

def test_forward_multigas():
    run = forward_fair({"emissions": rcp3pd.Emissions.emissions})
    assert run.concentration.shape == (736, 31)
    assert run.radiative_forcing.shape == (736, 13)

def test_inverse():
    nt = 141
    Cpi = 284.32
    C = Cpi * 1.01 ** np.arange(nt)

    tcr = 1.1
    ecs = 2.2

    config = {"tcr": tcr, "ecs": ecs, "C": C}
    run = inverse_fair(config)
    assert run.temperature.shape == (141,)
    assert 'time' in run.coords
    assert not 'ensemble_member' in run.coords

import numpy as np

from fair_spice.fair.models import inverse_fair
from fair_spice.ensemble import make_member

def basic_run():
    nt = 141
    Cpi = 284.32
    C = Cpi * 1.01 ** np.arange(nt)

    tcr = 1.1
    ecs = 2.2

    config = {"tcr": tcr, "ecs": ecs, "C": C}
    return inverse_fair(config)

def test_make_member():
    _run = basic_run()

    # test that make_member creates a new dimension
    run = make_member(_run)
    assert run.temperature.shape == (1, 141)
    assert run.temperature.dims == ('ensemble_member', 'time')
    assert 'time' in run.coords
    assert 'ensemble_member' not in run.coords

    # test that make_member sets the ensemble member number
    # and creates an ensemble_member coord
    run = make_member(_run, 4)
    assert run.temperature.shape == (1, 141)
    assert run.ensemble_member.data == [4]
    assert 'ensemble_member' in run.coords

def test_make_member_invariant():
    # test that marking a coordinate invariant doesn't expand it's dimension
    _run = basic_run()

    run = make_member(_run)
    assert run.C.shape == (1, 141)
    assert run.C.dims == ('ensemble_member', 'time')

    run = make_member(_run, invariant=['C'])
    assert run.C.shape == (141,)
    assert run.C.dims == ('time',)

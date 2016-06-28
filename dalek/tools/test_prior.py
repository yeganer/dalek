import numpy as np
import pytest
from dalek.tools.prior import (
        Prior, CheckPrior,
        ParameterValidPrior,
        ParameterValuePrior,
        LuminosityPrior,
        DefaultPrior,
        INVALID, DEFAULT)
from dalek.tools.base import Chain, Link

from astropy import units as u, constants as c


class Dummy(Link):
    outputs = ('dummy',)

    def calculate(self):
        return True


def setup_function(function):
    Prior.first = True
    Prior.inputs = ('parameters',)


def test_chain_prior():
    p1 = Prior()
    p2 = Prior()

    assert Chain(p1, p2).full(parameters={})[0] == DEFAULT


# S<SI ALWAYS
@pytest.fixture(
        scope='function', params=[
            (0.5, DEFAULT),
            (0.3, INVALID),
            (0.75, INVALID),
                ])
def idict(request):
    si = request.param[0]
    s = 0.4
    o = np.nan if s + si > 1 else 1 - s - si

    return {
            'parameters': {
                'model.abundances.o': o,
                'model.abundances.si': si,
                'model.abundances.s': s,
                },
            }, request.param[1]


def test_si_s_prior(idict):

    p1 = ParameterValidPrior()
    prior = ParameterValuePrior(
            ['model.abundances.si',
                'model.abundances.s'],
            lambda si, s: si > s)
    obtained = Chain(p1, prior).full(idict[0])[0]
    assert obtained == idict[1]

    Prior.reset()

    assert DefaultPrior().full(idict[0])[0] == idict[1]


def test_checkprior(idict):
    prior = Chain(DefaultPrior(), Chain(CheckPrior(), Dummy(), breakable=True))
    obtained = prior(idict[0])['dummy']
    expected = True if idict[1] == 0 else None
    assert obtained == expected


@pytest.mark.parametrize(
        ['tinner', 'vinner'],
        [
            (1e4, 1.4e4),
            (1e4, 1e4),
            ]
        )
def test_t_inner_prior(tinner, vinner):
    time = u.Quantity(10, 'd').cgs
    luminosity_requested = u.Quantity(1e43, 'erg/s').cgs
    tinner = u.Quantity(tinner, 'K')
    vinner = u.Quantity(vinner, 'km/s').cgs
    parameters = {
            'plasma.initial_t_inner': tinner,
            'model.structure.velocity.item0': vinner,
            }
    p = LuminosityPrior(luminosity_requested, time)
    luminosity = (
                    4 * np.pi * c.sigma_sb.cgs *
                    (vinner * time) ** 2 *
                    tinner ** 4).to('erg/s')
    expected = DEFAULT if luminosity > luminosity_requested else INVALID

    assert expected == p.full(parameters=parameters)[0]

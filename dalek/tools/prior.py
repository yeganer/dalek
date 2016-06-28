import numpy as np

from astropy import constants as c, units as u
from dalek.tools.base import Link, BreakChainException, Chain

INVALID = -np.inf
DEFAULT = 0.0


class Prior(Link):
    '''
    A very basic prior with the ability to break execution of a subchain.
    '''
    inputs = ('parameters',)
    outputs = ('logprior',)
    first = True

    def __init__(self):
        self.inputs = Prior.inputs
        if self.first:
            Prior.first = False
            Prior.inputs = ('parameters', 'logprior',)

    def calculate(self, parameters, logprior=DEFAULT):
        return self.logprior(parameters) + logprior

    def logprior(self, parameters):
        return DEFAULT

    @classmethod
    def reset(cls):
        cls.first = True
        cls.inputs = ('parameters',)


class DefaultPrior(Chain):

    def __init__(self):
        super(DefaultPrior, self).__init__(
                ParameterValidPrior(),
                ParameterValuePrior(
                    ['model.abundances.si',
                        'model.abundances.s'],
                    lambda si, s: si > s),
                ParameterValuePrior(
                    ['model.abundances.o'],
                    lambda o: 1 >= o >= 0),
                )


class ParameterValidPrior(Prior):

    def logprior(self, parameters):
        try:
            values = parameters.values()
        except TypeError:  # Dict object has .values()
            values = parameters.values
        if np.any(np.isnan(np.array(values))):
            return INVALID
        else:
            return DEFAULT


class ParameterValuePrior(Prior):

    def __init__(self, names, function):
        super(ParameterValuePrior, self).__init__()
        self._names = names
        self._fvalid = function

    def logprior(self, parameters):
        args = []
        for name in self._names:
            args.append(parameters[name])

        if self._fvalid(*args):
            return DEFAULT
        else:
            return INVALID


class LuminosityPrior(ParameterValuePrior):

    def __init__(self, luminosity_requested, time_explosion):
        def check_luminosity(tinner, vinner):
            #WARNING: Hardcoded quantities
            tinner = u.Quantity(tinner, 'K')
            vinner = u.Quantity(vinner, 'km/s').cgs
            luminosity = (
                    4 * np.pi * c.sigma_sb.cgs *
                    (vinner * time_explosion) ** 2 *
                    tinner ** 4).to('erg/s')
            return luminosity > luminosity_requested

        super(LuminosityPrior, self).__init__(
                names=[
                    'plasma.initial_t_inner',
                    'model.structure.velocity.item0'],
                function=check_luminosity
                )


class CheckPrior(Link):
    '''
    This will break the innermost dalek.tools.base.Chain to save computing time
    if the prior is INVALID.
    This should be the first element in the Chain containing the Tardis
    evaluation.
    '''
    inputs = ('logprior',)

    def calculate(self, logprior):
        if logprior == INVALID:
            raise BreakChainException(
                    'Prior is INVALID: {}'.format(logprior))

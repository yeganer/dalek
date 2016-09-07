from dalek.tools.base import Link

from specutils import extinction


class Extinction(Link):
    inputs = ('flux',)
    outputs = ('flux',)

    def __init__(self, wavelength, av, rv):
        self.extinction = extinction.extinction_ccm89(
                wavelength, av, rv)

    def calculate(self, flux):
        return flux * 10**(-0.4 * self.extinction)

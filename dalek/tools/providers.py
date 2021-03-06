import platform
import numpy as np
from uuid import uuid4
from astropy import units as u
from datetime import datetime as dt

from dalek.tools.base import Link


class PacketProvider(Link):
    inputs = ('model',)
    outputs = ('packet_nu', 'packet_energy',)

    def calculate(self, model):
        return (
                model.runner.emitted_packet_nu,
                model.runner.emitted_packet_luminosity,
                )


class VirtualPacketProvider(Link):
    inputs = ('model',)
    outputs = ('virtual_packet_nu', 'virtual_packet_energy',)

    def calculate(self, model):
        return (
                model.runner.virt_packet_nus * u.Hz,
                model.runner.virt_packet_energies /
                model.runner.time_of_simulation * u.erg,
                )


class Luminosity(Link):
    inputs = ('packet_nu', 'packet_energy',)
    outputs = ('luminosity',)

    def __init__(self, wl):
        self._wl_bins = wl

    def calculate(self, packet_nu, packet_energy):
        return (np.histogram(
            packet_nu.to(u.angstrom, u.spectral()),
            weights=packet_energy,
            bins=self._wl_bins)[0] * packet_energy.unit  / np.diff(self._wl_bins)
            )


class VirtualLuminosity(Luminosity):
    inputs = ('virtual_packet_nu', 'virtual_packet_energy',)

    def calculate(self, virtual_packet_nu, virtual_packet_energy):
        return super(
                VirtualLuminosity, self).calculate(
                        virtual_packet_nu, virtual_packet_energy)


class Flux(Link):
    inputs = ('luminosity',)
    outputs = ('flux',)

    def __init__(self, distance=(1 * u.Mpc)):
        self._distance = distance.to('cm')

    def calculate(self, luminosity):
        return luminosity / (4 * np.pi * self._distance**2)


class RunInfo(Link):
    inputs = tuple()
    outputs = ('iteration', 'rank', 'uuid', 'host', 'time')

    def __init__(self):
        self._iteration = 0
        try:
            from mpi4py import MPI
        except ImportError:
            self._rank = 0
        else:
            self._rank = MPI.COMM_WORLD.Get_rank()

    def calculate(self):
        self._iteration += 1
        return (
                self._iteration,
                self._rank,
                uuid4(),
                platform.node(),
                np.datetime64(dt.now()),
                )


class RunInfoFromFile(RunInfo):
    """
    Not yet modified to reflect new MetaContainer
    """

    def __init__(self, container):
        super(RunInfoFromFile, self).__init__()
        self._container = container

    def calculate(self):
        with self._container as store:
            try:
                df = store['run_table']
                df = df.loc[df.index.get_level_values('uid')==self._rank]
                iteration = df.index.get_level_values('iteration').max()
            except (KeyError, IndexError):
                pass
            else:
                if not np.isnan(iteration):
                    self._iteration = iteration
        return super(RunInfoFromFile, self).calculate()

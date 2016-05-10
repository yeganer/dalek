# import os
import time
import pytest
import tempfile
from uuid import uuid4
import numpy as np
import pandas as pd

from copy import copy

from dalek.base.meta import MetaContainer, MetaInformation, MPIContainer


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class DummyRadial1D(object):

    def __init__(self):
        class Spectrum(object):
            def __init__(self):
                self.luminosity_density_lambda = np.random.random(10000)
        self.t_rads = np.linspace(15000, 1000, 20)
        self.ws = np.random.random(20)
        self.spectrum = Spectrum()
        self.spectrum_virtual = Spectrum()


class DummyWrapper(object):

    def __init__(self):
        self.model = DummyRadial1D()
        self.uuid = uuid4()
        self.iteration = 1
        self.rank = 0
        self.fitness = 123


class TestMetaContainer(object):

    @classmethod
    @pytest.fixture(scope='class', autouse=True)
    def setup(self, request):
        self._file = tempfile.NamedTemporaryFile()
        self.container = MetaContainer(self._file.name)

    def test_save(self):
        tmp = pd.Series(np.arange(10))
        self.container.save(tmp, 'test')
        with pd.HDFStore(self._file.name, 'r') as store:
            assert np.all(tmp == store['test'].values)


class TestMetaInformation(object):

    @classmethod
    @pytest.fixture(scope='function', autouse=True)
    def setup(self, request):
        self._file = tempfile.NamedTemporaryFile()
        self.container = MetaContainer(self._file.name)

        # def fin():
        #     os.remove(self._file.name)
        # request.addfinalizer(fin)

    @pytest.fixture(scope='class')
    def parameter_dict(self):
        return {
                'model.abundances.o': 0.1,
                'raw.model.abundances.o': 0.01
                }

    @pytest.fixture(scope='function')
    def instance(self):
        return MetaInformation(
                np.random.randint(10),
                np.random.randint(100),
                np.random.random(),
                uuid4(),
                {})

    # TODO: Write test with hypothesis
    def test_init(self):
        uid = np.random.randint(10)
        iteration = np.random.randint(100)
        probability = np.random.random()
        name = uuid4()
        instance = MetaInformation(
                uid=uid,
                iteration=iteration,
                probability=probability,
                name=name,
                info_dict={})
        assert instance._uid == uid
        assert instance._iteration == iteration
        assert instance._probability == probability
        assert instance._name == name
        assert instance._info_dict == {}
        instance2 = MetaInformation(
                uid,
                iteration,
                probability,
                name,
                {})
        assert instance2._uid == uid
        assert instance2._iteration == iteration
        assert instance2._probability == probability
        assert instance2._name == name
        assert instance2._info_dict == {}

    # @pytest.mark.skipif(True, reason='debug')
    def test_save(self, parameter_dict, instance):
        instance._info_dict = parameter_dict
        self.container.save(instance, 'run_table')
        with pd.HDFStore(self._file.name, 'r') as store:
            assert np.all(
                    store['run_table'].loc[instance.at] ==
                    instance.df.loc[instance.at])

    # @pytest.mark.skipif(True, reason='debug')
    def test_save_multiple(self, parameter_dict, instance):
        instance._info_dict = parameter_dict
        self.container.save(instance, 'run_table')
        instance2 = copy(instance)
        instance2._iteration += 1
        self.container.save(instance2, 'run_table')
        with pd.HDFStore(self._file.name, 'r') as store:
            assert np.all(
                    store['run_table'].loc[instance.at] ==
                    instance.df.loc[instance.at])
            assert np.all(
                    store['run_table'].loc[instance2.at] ==
                    instance2.df.loc[instance2.at])

    def test_incremental_save(self):
        instance = MetaInformation(
                uid=np.random.randint(10),
                iteration=np.random.randint(100),
                probability=np.nan,
                name=uuid4(),
                info_dict={
                    'val1': 0.5,
                    'val2': np.nan,
                    })
        self.container.save(instance, 'run_table')
        instance._probability = np.random.random()
        instance._info_dict['val2'] = np.random.random()*100
        self.container.save(instance, 'run_table')
        with pd.HDFStore(self._file.name, 'r') as store:
            assert len(store['run_table']) == 1
            assert np.all(
                    store['run_table'].loc[instance.at] ==
                    instance.df.loc[instance.at])
        instance2 = copy(instance)
        instance2._iteration += 1
        self.container.save(instance2, 'run_table')
        with pd.HDFStore(self._file.name, 'r') as store:
            assert np.all(
                    store['run_table'].loc[instance2.at] ==
                    instance2.df.loc[instance2.at])


@pytest.mark.skipif(
        MPI == None or MPI.COMM_WORLD.size < 2,
        reason="No MPI support found")
def test_mpi_container():
    """
    Run MPI test.
    """
    comm = MPI.COMM_WORLD.Dup()
    rank = comm.rank
    f = tempfile.NamedTemporaryFile()
    mpi_container = MPIContainer(f.name, comm=comm)

    np.random.seed(12345)
    meta_info = [MetaInformation(
            uid=np.random.randint(10),
            iteration=np.random.randint(100),
            probability=np.random.random(),
            name='abc{}abc'.format(np.random.randint(10)),
            info_dict={
                'val1': 0.5,
                'val2': np.random.random(),
                }) for _ in range(comm.size)]
    for m in meta_info:
        m.add_data(pd.Series(np.arange(10)), 'foo')

    if rank == 0:
        mpi_container.receive()
        time.sleep(2)
        mpi_container.stop()
        mpi_container.join()
        with pd.HDFStore(f.name, 'r') as store:
            for m in meta_info[1:]:
                assert np.all(
                        store['test'].loc[m.at] ==
                        m.df.loc[m.at])
                assert np.all(
                        store[m.data_path('foo')] ==
                        m._foo)

    if rank > 0:
        print('sending data...')
        mpi_container.save(meta_info[rank], 'test')
        with pytest.raises(IOError):
            with pd.HDFStore(f.name, 'r') as store:
                    store['test']

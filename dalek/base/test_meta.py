import os
import pytest
import tempfile
from uuid import uuid4
import numpy as np
import pandas as pd

from copy import copy

from dalek.base.meta import MetaContainer, MetaInformation


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

        def fin():
            os.remove(self._file.name)
        request.addfinalizer(fin)

    def test_read_write(self):
        tmp = pd.Series(np.arange(10))
        with self.container as store:
            store['s'] = tmp
            print(store.filename)

        with self.container as store:
            print(store.filename)
            assert np.all(tmp == store['s'].values)
            del store['s']


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
                'o': 0.1,
                'o_raw': 0.01
                }

    @pytest.fixture(scope='function')
    def instance(self):
        return MetaInformation(
                np.random.randint(10),
                np.random.randint(100),
                np.random.random(),
                uuid4(),
                {})

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
        instance.save(self.container)
        with self.container as store:
            assert np.all(
                    store['run_table'].loc[instance.at] ==
                    instance.details.loc[instance.at])

    # @pytest.mark.skipif(True, reason='debug')
    def test_save_multiple(self, parameter_dict, instance):
        instance._info_dict = parameter_dict
        instance.save(self.container)
        instance2 = copy(instance)
        instance2._iteration = 1
        instance2.save(self.container)
        with self.container as store:
            assert np.all(
                    store['run_table'].loc[instance.at] ==
                    instance.details.loc[instance.at])
            assert np.all(
                    store['run_table'].loc[instance2.at] ==
                    instance2.details.loc[instance2.at])

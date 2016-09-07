import numpy as np
import pandas as pd

from dalek.tools.base import Link

from dalek.base.meta import MetaInformation


class SaveRun(Link):
    inputs = (
            'model', 'uuid', 'rank', 'iteration', 'parameters',
            'posterior', 'flux', 'runtime', 'host', 'time')
    outputs = tuple()

    def __init__(
            self, container=None, add_data=[],
            table_name='run_table', flux=False):
        if container is None:
            self.calculate = lambda *args, **kwargs: None
        self._container = container
        self._table_name = table_name
        self._add_data = add_data
        self._flux = flux

    def calculate(
            self, model, uuid, rank, iteration, parameters,
            posterior=np.nan, flux=np.nan, runtime=np.nan, host=None,
            time=None):
        try:
            run_values = parameters.as_full_dict()
        except AttributeError:
            run_values = parameters

        run_values['runtime'] = runtime or 0.0
        run_values['start_time'] = time or np.datetime64()
        run_values['host'] = host or 'unknown'

        metainfo = MetaInformation(
                uid=rank,
                iteration=iteration,
                probability=posterior,
                name=uuid,
                info_dict=run_values,
                )
        if self._flux and flux and not np.any(np.isnan(flux)):
            metainfo.add_data(pd.Series(flux), 'flux')
        for name in self._add_data:
            obj = model
            try:
                for p in name.split('.'):
                    obj = getattr(obj, p)
            except (AttributeError, KeyError):
                # TODO: proper error handling here
                pass
            else:
                name = name.replace('.', '_')
                metainfo.add_data(pd.Series(obj), name)
        self._container.save(metainfo, self._table_name)


class SavePreRun(SaveRun):
    '''
    Test missing!!!
    '''
    inputs = (
            'uuid', 'rank', 'iteration', 'parameters',
            'host', 'time')

    def calculate(
            self, uuid, rank, iteration, parameters,
            host, time):
        super(SavePreRun, self).calculate(
                None, uuid, rank, iteration, parameters, host=host, time=time)

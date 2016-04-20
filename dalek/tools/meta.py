import pandas as pd

from dalek.tools.base import Link

from dalek.base.meta import MetaInformation

class SaveRun(Link):
    inputs = ('model', 'uuid', 'rank', 'iteration', 'posterior', 'runtime', 'parameters')
    outputs = tuple()

    def __init__(self, container, add_data=[]):
        self._container = container
        self._add_data = add_data

    def calculate(self, model, uuid, rank, iteration, posterior, runtime, parameters):
        try:
            run_values = parameters.as_full_dict()
        except AttributeError:
            run_values = parameters

        run_values['runtime'] = runtime or 0.0

        metainfo = MetaInformation(
                uid=rank,
                iteration=iteration,
                probability=posterior,
                name=uuid,
                info_dict=run_values,
                table_name='run_table',
                )
        for name in self._add_data:
            try:
                metainfo.add_data(name, pd.Series( getattr(model, name)))
            except (AttributeError, KeyError):
                pass
        metainfo.save(self._container)

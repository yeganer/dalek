# coding: utf-8
import os
import pandas as pd
from dalek.wrapper import SafeHDFStore
import warnings
import tables


class MetaContainer(object):
    """
    Class to control storing of metainformation

    Parameters
    -----
        path: string
            Absolute path to output .h5 file
        summary_data: dictionary-like object
            Additional information that should be saved.
            For example dalek_version, tardis_version, time,
            number of ranks, iterations

    """
    def __init__(self, path, summary_data=None):
        self._path, self._file = os.path.split(path)
        self.open = False
        if summary_data is not None:
            self._write_summary(summary_data)

    def _write_summary(self, data):
        '''
        Write summary_data to .h5 file.
        '''
        # TODO: proper dictionary to pd.Series parsing
        with self as store:
            if 'overview' not in store.keys():
                pd.Series(
                        data.values(), index=data.keys()
                        ).to_hdf(store, 'overview')

    def __enter__(self):
        '''
        Wrapper function to allow

        with MetaContainer as store:
            <code>
        '''
        if not self.open:
            self.open = True
            self.store = SafeHDFStore(os.path.join(self._path, self._file))
        return self.store

    def __exit__(self, type, value, traceback):
        '''
        close store after 'with'
        '''
        if self.open:
            self.store.__exit__(type, value, traceback)
            self.open = False
        return


class MetaInformation(object):
    '''
    A collection of information on one particular run.

    Parameters:
        int uid: unique ID of walker/population candidate etc.
        int iteration: how many iteration were done with this ID
        float probability: probability of this run
        str name: currently uuid4() is used to name individual runs
        dict info_dict: dictionary of arbitary key/value paires to be
                            appended to the table
    '''

    def __init__(self, uid, iteration, probability, name,
                 info_dict=None, table_name='run_table'):
        self._uid = uid
        self._iteration = iteration
        self._probability = probability
        self._name = name
        self._info_dict = info_dict
        self._data = list()
        self._table_name = table_name

    def add_data(self, name, data):
        self._data.append(name)
        if hasattr(self, '_' + name):
            raise ValueError(
                    '''MetaInformation already has attribute _{:s}.
Please choose another name.\n'''.format(name))
        else:
            setattr(self, '_' + name, data)

    def data_path(self, name=''):
        return 'data/{}/{}'.format(self._name, name)

    @property
    def run_details(self):
        res = {
                'uid': self._uid,
                'iteration': self._iteration,
                'name': str(self._name),
                'probability': self._probability,
                }
        try:
            res.update(self._info_dict)
        except AttributeError:
            pass
        return res

    @property
    def details(self):
        return (
                pd.DataFrame.from_records([self.run_details])
                .set_index(['iteration', 'uid']))

    def _save_data(self, store):
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for d in self._data:
            try:
                store[self.data_path(d)]
            except KeyError:
                getattr(self, '_' + d).to_hdf(store, self.data_path(d))

    def save(self, container):
        with container as store:
            try:
                if self.at in store[self._table_name].index:
                    df = store[self._table_name]
                    del store[self._table_name]
                    df.loc[self.at] = self.details.loc[self.at]
                else:
                    df = self.details
            except KeyError:
                df = self.details
            store.append(self._table_name, df)
            self._save_data(store)

    @property
    def at(self):
        return self._iteration, self._uid

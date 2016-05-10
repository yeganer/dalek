# coding: utf-8
import os
import time
import tables
import warnings
import pandas as pd
# from multiprocessing import Process
from threading import Thread

from dalek.wrapper import SafeHDFStore


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class _end_receive_message(object):

    def __repr__(self):
        return '<End receiving message>'


class _item_wrapper(object):

    def __init__(self, item, path):
        self.item = item
        self.path = path

    def __repr__(self):
        return ', '.join([str(self.item), self.path])


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
    def __init__(self, path, summary_data=None, comm=None):
        self._path, self._file = os.path.split(path)
        if summary_data is not None:
            self._write_summary(summary_data)

    def _write_summary(self, data):
        '''
        Write summary_data to .h5 file.
        '''
        # TODO: proper dictionary to pd.Series parsing
        with SafeHDFStore(os.path.join(self._path, self._file)) as store:
            if 'overview' not in store.keys():
                pd.Series(
                        data.values(), index=data.keys()
                        ).to_hdf(store, 'overview')

    def save(self, item, path):
        with SafeHDFStore(os.path.join(self._path, self._file)) as store:
            item.to_hdf(store, path)


class MPIContainer(object):

    def __init__(self, *args, **kwargs):
        comm = kwargs.pop('comm', None)
        self.comm = MPI.COMM_WORLD.Dup() if comm is None else comm
        self.debug = kwargs.pop('debug', False)
        self.status = MPI.Status()
        self.container = MetaContainer(*args, **kwargs)
        self._p = None

    def is_master(self):
        return self.comm.rank == 0

    def save(self, item, path):
        if self.debug:
            print('save: {} to {}'.format(item, path))

        if self.is_master():
            self.container.save(item, path)
        else:
            if self.debug:
                print('save: {} sending data to master'.format(
                    self.comm.rank))
            self.comm.send(
                    _item_wrapper(item, path),
                    dest=0,
                    # tag=4,
                    tag=self.comm.rank
                    )

    def receive(self):
        assert self.is_master()
        # Define function that will run in a seperate thread
        def f():
            while True:
                # Waiting for a message
                if self.debug:
                    print('receive: waiting...')

                # Receive a message
                data = self.comm.recv(
                        source=MPI.ANY_SOURCE,
                        tag=MPI.ANY_TAG,
                        status=self.status)

                if self.debug:
                    print('receive: from {} with tag {}'.format(
                        self.status.source, self.status.tag))
                    print('receive: data is {}'.format(data))

                # Stop the loop on special object
                if (isinstance(data, _end_receive_message) and
                        self.status.source == 0):
                    if self.debug:
                        print("receive: end signal. stopping")
                    break

                # Save the data
                self.container.save(data.item, data.path)
                if self.debug:
                    print('receive: finished saving {}'.format(data.item))

                # Wait a small amount, might be unnecessary
                time.sleep(0.1)
            return

        # Create and start the thread
        self._p = Thread(target=f)
        self._p.start()

    def join(self):
        try:
            self._p.join()
        except AttributeError:
            pass

    def stop(self):
        assert self.is_master()
        if self._p and self._p.is_alive():
            self.comm.send(
                    _end_receive_message(),
                    dest=0,
                    tag=0
                    )

        self.join()
        self._p = None

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
                 info_dict={}):
        self._uid = uid
        self._iteration = iteration
        self._probability = probability
        self._name = name
        self._info_dict = info_dict
        self._data = set()

    def __repr__(self):
        return '<MetaInformation name {} at {}>'.format(self._name, self.at)

    def add_data(self, data, name):
        if hasattr(self, '_' + name):
            raise ValueError(
                    '''MetaInformation already has attribute _{:s}.
Please choose another name.\n'''.format(name))
        else:
            setattr(self, '_' + name, data)
        self._data.add(name)

    def data_path(self, dataname=''):
        return 'data/{}/{}'.format(self._name, dataname)

    @property
    def df(self):
        res = {
                'uid': self._uid,
                'iteration': self._iteration,
                'name': str(self._name),
                'probability': self._probability,
                }
        res.update(self._info_dict)
        return (
                pd.DataFrame.from_records([res])
                .set_index(['iteration', 'uid']))

    def _save_data(self, store):
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for d in self._data:
            try:
                store[self.data_path(d)]
            except KeyError:
                getattr(self, '_' + d).to_hdf(store, self.data_path(d))

    def to_hdf(self, store, path):
        # print('to_hdf: uid = {}'.format(self._uid))
        try:
            if self.at in store[path].index:
                df = store[path]
                del store[path]
                df.loc[self.at] = self.df.loc[self.at]
            else:
                df = self.df
        except KeyError:
            df = self.df
        store.append(path, df)
        self._save_data(store)

    @property
    def at(self):
        return self._iteration, self._uid

import os
import time
import errno
import logging

from pandas import HDFStore

logger = logging.getLogger(__name__)


class SafeHDFStore(HDFStore):
    def open(self, *args, **kwargs):
        probe_interval = kwargs.pop("probe_interval", 0.1)
        self._lock = "%s.lock" % self._path
        while True:
            try:
                self._flock = os.open(
                        self._lock, os.O_CREAT |
                        os.O_EXCL |
                        os.O_WRONLY)
                break
            except OSError as e:
                if e.errno == errno.EEXIST:
                    time.sleep(probe_interval)
                else:
                    raise e

        super(SafeHDFStore, self).open(*args, **kwargs)

    def close(self):
        logger.debug('Close SafeHDFStore')
        super(SafeHDFStore, self).close()
        os.close(self._flock)
        os.remove(self._lock)

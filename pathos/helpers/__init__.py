#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/pathos/LICENSE

import pp_helper
import mp_helper
import pp as parallelpython

try:
    import processing as mp
    from processing.pool import Pool as ProcessPool  # use pathos/external
    from processing import cpuCount as cpu_count
    import Queue

    class ThreadPool(ProcessPool):
        from processing.dummy import Process
        def __init__(self, processes=None, initializer=None, initargs=()):
            ProcessPool.__init__(self, processes, initializer, initargs)
            return
        def _setup_queues(self):
            self._inqueue = Queue.Queue()
            self._outqueue = Queue.Queue()
            self._quick_put = self._inqueue.put
            self._quick_get = self._outqueue.get
            return
        @staticmethod
        def _help_stuff_finish(inqueue, task_handler, size):
            # put sentinels at head of inqueue to make workers finish
            inqueue.not_empty.acquire()
            try:
                inqueue.queue.clear()
                inqueue.queue.extend([None] * size)
                inqueue.not_empty.notify_all()
            finally:
                inqueue.not_empty.release()
            return

except ImportError:  # fall-back to package distributed with python
    import multiprocessing as mp
    from multiprocessing.pool import Pool as ProcessPool
    from multiprocessing import cpu_count
    from multiprocessing.dummy import Pool as ThreadPool


#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/pathos/LICENSE
"""
This module contains map and pipe interfaces to python's multiprocessing module.

Pipe methods provided:
    pipe        - blocking communication pipe             [returns: value]
    apipe       - asynchronous communication pipe         [returns: object]

Map methods provided:
    map         - blocking and ordered worker pool        [returns: list]
    imap        - non-blocking and ordered worker pool    [returns: iterator]
    uimap       - non-blocking and unordered worker pool  [returns: iterator]
    amap        - asynchronous worker pool                [returns: object]


Usage
=====

A typical call to a pathos multiprocessing map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.multiprocessing import ProcessingPool
    >>> pool = ProcessingPool(nodes=4)
    >>>
    >>> # do a blocking map on the chosen function
    >>> print pool.map(pow, [1,2,3,4], [5,6,7,8])
    >>>
    >>> # do a non-blocking map, then extract the results from the iterator
    >>> results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    >>> print "..."
    >>> print list(results)
    >>>
    >>> # do an asynchronous map, then get the results
    >>> results = pool.amap(pow, [1,2,3,4], [5,6,7,8])
    >>> while not results.ready():
    >>>     time.sleep(5); print ".",
    >>> print results.get()
    >>>
    >>> # do one item at a time, using a pipe
    >>> print pool.pipe(pow, 1, 5)
    >>> print pool.pipe(pow, 2, 6)
    >>>
    >>> # do one item at a time, using an asynchronous pipe
    >>> result1 = pool.apipe(pow, 1, 5)
    >>> result2 = pool.apipe(pow, 2, 6)
    >>> print result1.get()
    >>> print result2.get()


Notes
=====

This worker pool leverages the python's multiprocessing module, and thus
has many of the limitations associated with that module. The function f and
the sequences in args must be serializable. The maps in this worker pool
have full functionality whether run from a script or in the python
interpreter, and work reliably for both imported and interactively-defined
functions. Unlike python's multiprocessing module, pathos.multiprocessing maps
can directly utilize functions that require multiple arguments.

"""
__all__ = ['ProcessingPool','ThreadingPool']

#FIXME: probably not good enough... should store each instance with a uid
__STATE = _ProcessingPool__STATE = \
          _ThreadingPool__STATE = {'pool':None, 'threads':None}

from pathos.abstract_launcher import AbstractWorkerPool
from pathos.helpers.mp_helper import starargs as star
from pathos.helpers import ProcessPool as Pool
from pathos.helpers import cpu_count, ThreadPool

class ProcessingPool(AbstractWorkerPool):
    """
Mapper that leverages python's multiprocessing.
    """
    def __init__(self, *args, **kwds):
        """\nNOTE: if number of nodes is not given, will autodetect processors
        """
        hasnodes = kwds.has_key('nodes'); arglen = len(args)
        if kwds.has_key('ncpus') and (hasnodes or arglen):
            msg = "got multiple values for keyword argument 'ncpus'"
            raise TypeError, msg
        elif hasnodes: #XXX: multiple try/except is faster?
            if arglen:
                msg = "got multiple values for keyword argument 'nodes'"
                raise TypeError, msg
            kwds['ncpus'] = kwds.pop('nodes')
        elif arglen:
            kwds['ncpus'] = args[0]
        self.__nodes = kwds.get('ncpus', cpu_count())

        # Create a new server if one isn't already initialized
        self._serve()
        return
    __init__.__doc__ = AbstractWorkerPool.__init__.__doc__ + __init__.__doc__
   #def __exit__(self, *args):
   #    self._clear()
   #    return
    def _serve(self, nodes=None): #XXX: should be STATE method; use id
        """Create a new server if one isn't already initialized"""
        if nodes is None: nodes = self.__nodes
        _pool = __STATE['pool']
        if not _pool or nodes != _pool.__nodes:
            _pool = Pool(nodes)
            _pool.__nodes = nodes
            __STATE['pool'] = _pool
        return _pool
    def _clear(self): #XXX: should be STATE method; use id
        """Remove server with matching state"""
        _pool = __STATE['pool']
        if _pool and self.__nodes == _pool.__nodes:
            __STATE['pool'] = None
        return #_pool
    def map(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.map(star(f), zip(*args)) # chunksize
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.imap(star(f), zip(*args)) # chunksize
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    def uimap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.imap_unordered(star(f), zip(*args)) # chunksize
    uimap.__doc__ = AbstractWorkerPool.uimap.__doc__
    def amap(self, f, *args, **kwds): # register a callback ?
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.map_async(star(f), zip(*args)) # chunksize
    amap.__doc__ = AbstractWorkerPool.amap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.apply(f, args, kwds)
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    def apipe(self, f, *args, **kwds): # register a callback ?
       #AbstractWorkerPool._AbstractWorkerPool__apipe(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.applyAsync(f, args, kwds)
    apipe.__doc__ = AbstractWorkerPool.apipe.__doc__
    ########################################################################
    def __repr__(self):
        mapargs = (self.__class__.__name__, self.ncpus)
        return "<pool %s(ncpus=%s)>" % mapargs
    def __get_nodes(self):
        """get the number of nodes used in the map"""
        return self.__nodes
    def __set_nodes(self, nodes):
        """set the number of nodes used in the map"""
        self._serve(nodes)
        self.__nodes = nodes
        return
    # interface
    ncpus = property(__get_nodes, __set_nodes)
    nodes = property(__get_nodes, __set_nodes)
    pass


class ThreadingPool(AbstractWorkerPool):
    """
Mapper that leverages python's threading.
    """
    def __init__(self, *args, **kwds):
        """\nNOTE: if number of nodes is not given, will autodetect processors
        """
        hasnodes = kwds.has_key('nodes'); arglen = len(args)
        if kwds.has_key('nthreads') and (hasnodes or arglen):
            msg = "got multiple values for keyword argument 'nthreads'"
            raise TypeError, msg
        elif hasnodes: #XXX: multiple try/except is faster?
            if arglen:
                msg = "got multiple values for keyword argument 'nodes'"
                raise TypeError, msg
            kwds['nthreads'] = kwds.pop('nodes')
        elif arglen:
            kwds['nthreads'] = args[0]
        self.__nodes = kwds.get('nthreads', cpu_count())

        # Create a new server if one isn't already initialized
        self._serve()
        return
    __init__.__doc__ = AbstractWorkerPool.__init__.__doc__ + __init__.__doc__
   #def __exit__(self, *args):
   #    self._clear()
   #    return
    def _serve(self, nodes=None): #XXX: should be STATE method; use id
        """Create a new server if one isn't already initialized"""
        if nodes is None: nodes = self.__nodes
        _pool = __STATE['threads']
        if not _pool or nodes != _pool.__nodes:
            _pool = ThreadPool(nodes)
            _pool.__nodes = nodes
            __STATE['threads'] = _pool
        return _pool
    def _clear(self): #XXX: should be STATE method; use id
        """Remove server with matching state"""
        _pool = __STATE['threads']
        if _pool and self.__nodes == _pool.__nodes:
            __STATE['threads'] = None
        return #_pool
    def map(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.map(star(f), zip(*args)) # chunksize
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.imap(star(f), zip(*args)) # chunksize
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    def uimap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.imap_unordered(star(f), zip(*args)) # chunksize
    uimap.__doc__ = AbstractWorkerPool.uimap.__doc__
    def amap(self, f, *args, **kwds): # register a callback ?
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.map_async(star(f), zip(*args)) # chunksize
    amap.__doc__ = AbstractWorkerPool.amap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.apply(f, args, kwds)
   #pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    def apipe(self, f, *args, **kwds): # register a callback ?
       #AbstractWorkerPool._AbstractWorkerPool__apipe(self, f, *args, **kwds)
        _pool = self._serve()
        return _pool.applyAsync(f, args, kwds)
   #apipe.__doc__ = AbstractWorkerPool.apipe.__doc__
    ########################################################################
    def __repr__(self):
        mapargs = (self.__class__.__name__, self.nthreads)
        return "<pool %s(nthreads=%s)>" % mapargs
    def __get_nodes(self):
        """get the number of nodes used in the map"""
        return self.__nodes
    def __set_nodes(self, nodes):
        """set the number of nodes used in the map"""
        self._serve(nodes)
        self.__nodes = nodes
        return
    # interface
    nthreads = property(__get_nodes, __set_nodes)
    nodes = property(__get_nodes, __set_nodes)
    pass



#!/usr/bin/env python
#
# Based on code by Kirk Strauser <kirk@strauser.com>
# Rev: 1139; Date: 2008-04-16
# (see license text in pathos.pp_map)
#
# Forked by: Mike McKerns (April 2008)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/pathos/LICENSE
#
# Modified to meet the pathos pool API
"""
This module contains map and pipe interfaces to the parallelpython (pp) module.

Pipe methods provided:
    pipe        - blocking communication pipe             [returns: value]
    apipe       - asynchronous communication pipe         [returns: object]

Map methods provided:
    map         - blocking and ordered worker pool        [returns: list]
    imap        - non-blocking and ordered worker pool    [returns: iterator]
    amap        - asynchronous worker pool                [returns: object]


Usage
=====

A typical call to a pathos pp map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.pp import ParallelPythonPool
    >>> pool = ParallelPythonPool(nodes=4)
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

This worker pool leverages the parallelpython (pp) module, and thus
has many of the limitations associated with that module. The function f and
the sequences in args must be serializable. The maps in this worker pool
have full functionality when run from a script, but may be somewhat limited
when used in the python interpreter. Both imported and interactively-defined
functions in the interpreter session may fail due to the pool failing to
find the source code for the target function. For a work-around, try:

    >>> # instantiate and configure the worker pool
    >>> from pathos.pp import ParallelPythonPool
    >>> pool = ParallelPythonPool(nodes=4)
    >>>
    >>> # wrap the function, so it can be used interactively by the pool
    >>> def wrapsin(*args, **kwds):
    >>>      from math import sin
    >>>      return sin(*args, **kwds)
    >>>
    >>> # do a blocking map using the wrapped function
    >>> results = pool.map(wrapsin, [1,2,3,4,5])

"""
__all__ = ['ParallelPythonPool', 'stats']

import __builtin__
from pathos.helpers import parallelpython as pp
from pathos.helpers import cpu_count

#FIXME: probably not good enough... should store each instance with a uid
__STATE = _ParallelPythonPool__STATE = {'server':None}

def __print_stats():
    "print stats from the pp.Server"
    if __STATE['server']:
        __STATE['server'].print_stats()
    else:
        print "Stats are not available; no active servers.\n"

def stats():  #XXX: better return object(?) to query? | is per run? compound?
    "return stats print string from the pp.Server"
    import StringIO, sys
    stdout = sys.stdout
    try:
        sys.stdout = result = StringIO.StringIO()
        __print_stats()
    except:
        result = None #FIXME: will cause error below
    sys.stdout = stdout
    result = result.getvalue()
    return result


from pathos.abstract_launcher import AbstractWorkerPool
from pathos.helpers.pp_helper import ApplyResult, MapResult

#XXX: should look into parallelpython for 'cluster computing'
class ParallelPythonPool(AbstractWorkerPool):
    """
Mapper that leverages parallelpython (i.e. pp) maps.
    """
    def __init__(self, *args, **kwds):
        """\nNOTE: if number of nodes is not given, will autodetect processors
NOTE: if a tuple of servers is not provided, defaults to localhost only
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
        self.__nodes = None
        self.__servers = ()

        ncpus = kwds.get('ncpus', None)
       #servers = kwds.get('servers', ('*',)) # autodetect
        servers = kwds.get('servers', ()) # only localhost
        if servers is None: servers = ()
       #from _ppserver_config import ppservers as servers # config file

        #XXX: throws 'socket.error' when starting > 1 server with autodetect
        # Create a new server if one isn't already initialized
        # ...and set the requested level of multi-processing
        _pool = self._serve(nodes=ncpus, servers=servers)
        #XXX: or register new UID for each instance?
        #_pool.set_ncpus(ncpus or 'autodetect') # no ncpus=0
       #print "configure", _pool.get_ncpus(), "local workers"
        return
    __init__.__doc__ = AbstractWorkerPool.__init__.__doc__ + __init__.__doc__
   #def __exit__(self, *args):
   #    self._clear()
   #    return
    def _serve(self, nodes=None, servers=None): #XXX: is a STATE method; use id
        """Create a new server if one isn't already initialized""" 
        # get nodes and servers in form used by pp.Server
        if nodes is None: nodes = self.nodes #XXX: autodetect must be explicit
        if nodes in ['*']: nodes = 'autodetect'
        if servers is None:
            servers = tuple(sorted(self.__servers)) # no servers is ()
        elif servers in ['*', 'autodetect']: servers = ('*',)
        # if no server, create one
        _pool = __STATE['server']
        if not _pool:
            _pool = pp.Server(ppservers=servers)
        # convert to form returned by pp.Server, then compare
        _auto = [('*',)] if _pool.auto_ppservers else []
        _servers = sorted(_pool.ppservers + _auto)
        _servers = tuple(':'.join((str(i) for i in tup)) for tup in _servers)
        if servers != _servers: #XXX: assume servers specifies ports if desired
            _pool = pp.Server(ppservers=servers)
        # convert to form returned by pp.Server, then compare
        _nodes = cpu_count() if nodes=='autodetect' else nodes
        if _nodes != _pool.get_ncpus():
            _pool.set_ncpus(nodes) # allows ncpus=0
        # set (or 'repoint') the server
        __STATE['server'] = _pool
        # set the 'self' internals
        self.__nodes = None if nodes in ['autodetect'] else nodes
        self.__servers = servers
        return _pool
    def _clear(self): #XXX: should be STATE method; use id
        """Remove server with matching state"""
        _pool = __STATE['server']
        if not _pool:
            return
        # convert to form returned by pp.Server, then compare
        _nodes = cpu_count() if self.__nodes is None else self.__nodes
        if _nodes != _pool.get_ncpus():
            return
        _auto = [('*',)] if _pool.auto_ppservers else []
        _servers = sorted(_pool.ppservers + _auto)
        _servers = [':'.join((str(i) for i in tup)) for tup in _servers]
        if sorted(self.__servers) != _servers:
            return
        # it's the 'same' (better to check _pool.secret?)
        __STATE['server'] = None
        return #_pool
    def map(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        return list(self.imap(f, *args))
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        def submit(*argz):
            """send a job to the server"""
            _pool = self._serve()
           #print "using", _pool.get_ncpus(), 'local workers'
            return _pool.submit(f, argz, globals=globals())
        # submit all jobs, then collect results as they become available
        return (subproc() for subproc in __builtin__.map(submit, *args))
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    def amap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        def submit(*argz):
            """send a job to the server"""
            _pool = self._serve()
           #print "using", _pool.get_ncpus(), 'local workers'
            return _pool.submit(f, argz, globals=globals())
        override = True if kwds.has_key('size') else False
        elem_size = kwds.pop('size', 2)
        args = zip(*args)
        # submit all jobs, to be collected later with 'get()'
        tasks = [submit(*task) for task in args]
        tasks = [ApplyResult(task) for task in tasks]
        # build a correctly sized results object
        length = len(args)
        nodes = self.nodes
        if self.nodes in ['*','autodetect',None]:
            _pool = self._serve()
            nodes = _pool.get_ncpus() #XXX: local workers only?
        # try to quickly find a small chunksize that gives good results
        maxsize = 2**62 #XXX: HOPEFULLY, this will never be reached...
        chunksize = 1
        while chunksize < maxsize:
            chunksize, extra = divmod(length, nodes * elem_size)
            if override: break # the user *wants* to override this loop
            if extra >= length: break # we found something that 'works'
            elem_size = elem_size * 2
        if extra: chunksize += 1
        m = MapResult((chunksize,length))
        # queue the tasks
        m.queue(*tasks)
        return m
    amap.__doc__ = AbstractWorkerPool.amap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        # submit a job to the server, and block until results are collected
        _pool = self._serve()
        task = _pool.submit(f, args, globals=globals())
        return task()
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    def apipe(self, f, *args, **kwds): # register a callback ?
       #AbstractWorkerPool._AbstractWorkerPool__apipe(self, f, *args, **kwds)
        # submit a job, to be collected later with 'get()'
        _pool = self._serve()
        task = _pool.submit(f, args, globals=globals())
        return ApplyResult(task)
    apipe.__doc__ = AbstractWorkerPool.apipe.__doc__
    ########################################################################
    def __repr__(self):
        mapargs = (self.__class__.__name__, self.ncpus, self.servers)
        return "<pool %s(ncpus=%s, servers=%s)>" % mapargs
    def __get_nodes(self):
        """get the number of nodes used in the map"""
        nodes = self.__nodes
        if nodes == None: nodes = '*'
        return nodes
    def __set_nodes(self, nodes):
        """set the number of nodes used in the map"""
        if nodes is None: nodes = 'autodetect'
        self._serve(nodes=nodes)
        return
    def __get_servers(self):
        """get the servers used in the map"""
        servers = self.__servers
        if servers == (): servers = None
        elif servers == ('*',): servers = '*'
        return servers
    def __set_servers(self, servers):
        """set the servers used in the map"""
        if servers is None: servers = ()
        self._serve(servers=servers)
        #__STATE['server'].ppservers == [(s.split(':')[0],int(s.split(':')[1])) for s in servers]
        # we could check if the above is true... for now we will just be lazy
        # we could also convert lists to tuples... again, we'll be lazy
        # XXX: throws "socket error" when autodiscovery service is enabled
        return
    # interface
    ncpus = property(__get_nodes, __set_nodes)
    nodes = property(__get_nodes, __set_nodes)
    servers = property(__get_servers, __set_servers)
    pass



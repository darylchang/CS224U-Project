#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/pathos/LICENSE
#
# adapted from J. Kim & M. McKerns utility functions
"""
utilities for distributed computing
"""

import os


def print_exc_info():
    """thread-safe return of string from print_exception call"""

    import StringIO, traceback
    
    sio = StringIO.StringIO()
    traceback.print_exc(file=sio) #thread-safe print_exception to string
    sio.seek(0, 0)
    
    return sio.read()


def spawn(onParent, onChild):
    """a fork wrapper

Calls onParent(pid, fromchild) in parent process,
      onChild(pid, toparent) in child process.
    """
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(c2pwrite)            
        fromchild = os.fdopen(c2pread)
        return onParent(pid, fromchild)

    os.close(c2pread)
    toparent = os.fdopen(c2pwrite, 'w', 0)
    pid = os.getpid()

    return onChild(pid, toparent)


def spawn2(onParent, onChild):
    """an alternate fork wrapper

Calls onParent(pid, fromchild, tochild) in parent process,
      onChild(pid, fromparent, toparent) in child process.
    """

    p2cread, p2cwrite = os.pipe()
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(p2cread)
        os.close(c2pwrite)            
        fromchild = os.fdopen(c2pread, 'r')
        tochild = os.fdopen(p2cwrite, 'w', 0)
        return onParent(pid, fromchild, tochild)

    os.close(p2cwrite)
    os.close(c2pread)
    fromparent = os.fdopen(p2cread, 'r')
    toparent = os.fdopen(c2pwrite, 'w', 0)
    pid = os.getpid()

    return onChild(pid, fromparent, toparent)


# End of file

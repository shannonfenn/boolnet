# cython: profile=False
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution)
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def connected_sources(np.ndarray[np.uint32_t, ndim=2] gates, 
                      np.ndarray[np.uint8_t, ndim=1] connected,
                      unsigned int Ni, unsigned int No):
    ''' This detects which gates and inputs are connected to the output
        and returns the gate indices as a sorted list. This
        is just the union of the connected components in the
        digraph where connections are only backward.'''
    cdef unsigned int Ng = gates.shape[0]
    cdef int g
    cdef unsigned int total = 0

    # The outputs are connected
    for g in range(Ng-No+Ni, Ng+Ni):
        connected[g] = 1

    # visit in reverse order
    # this only works due to the network being a topologically ordered DAG
    for g in reversed(range(Ng)):
        if connected[g + Ni]:
            connected[gates[g, 0]] = 1
            connected[gates[g, 1]] = 1
            total += 1

    # find and return non-zero indices
    cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(total, dtype=np.uint32)
    cdef unsigned int i = 0
    for g in range(Ni, Ni + Ng): # ignore connected inputs
        if connected[g]:
            result[i] = g - Ni
            i += 1

    return result

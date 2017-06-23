from boolnet.exptools.fastrand cimport random_uniform_int
import networkx as nx
import numpy as np
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef size_t sample_bool(np.uint8_t[:] M, size_t end=0):
    cdef size_t i, r, random_index, total
    total = 0
    
    if end == 0:
        end = M.size

    # count ones
    for i in range(end):
        total += M[i]
    
    # pick random position
    random_index = random_uniform_int(total)
    
    # find the '1' that is random_index positions from the start
    r = 0
    for i in range(end):
        # only check if this position is '1'
        if M[i]:
            if r == random_index:
                return i
            r += M[i]
    # occurs only when M.size = 0 or total = 0
    return M.size


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#cdef size_t sample_masked_bool(np.uint8_t[:] M, np.uint8_t[:] mask, size_t end=0):
#    cdef size_t i, r, random_index, total
#    total = 0
#    if end == 0:
#        end = min(M.size, mask.size)
#    for i in range(end):
#        total += M[i] & mask[i]
#    random_index = random_uniform_int(total)
#    r = 0
#    for i in range(end):
#        if M[i] & mask[i]:
#            if r == random_index:
#                return i
#            r += M[i] & mask[i]
#    return M.size


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef connected_sources(np.uint32_t[:, :] gates, np.uint8_t[:] connected,
                        size_t Ni, size_t No):
    ''' This detects which gates and inputs are connected to the output
        and returns the gate indices as a sorted list. This
        is just the union of the connected components in the
        digraph where connections are only backward.'''
    cdef:
        size_t Ng, total
        int g

    Ng = gates.shape[0]
    
    # The outputs are connected
    for g in range(Ng-No+Ni, Ng+Ni):
        connected[g] = 1

    total = No

    # visit in reverse order
    # this only works due to the network being a topologically ordered DAG
    for g in reversed(range(Ng)):
        if connected[g + Ni]:
            connected[gates[g, 0]] = 1
            connected[gates[g, 1]] = 1


cpdef filter_connected(np.uint32_t[:, :] gates, size_t Ni, size_t No):
    Ng = gates.shape[0]
    connected = np.zeros(Ng + Ni, dtype=np.uint8)
    connected_sources(gates, connected, Ni, No)

    new_gates = []
    source_map = np.arange(Ni + Ng)
    for old_index in range(Ni, Ni + Ng):
        if connected[old_index]:
            source_map[old_index] = len(new_gates) + Ni
            old_gate = gates[old_index - Ni]
            # remapped connections
            new_gate = [source_map[s] for s in old_gate[:-1]]
            # transfer function
            new_gate.append(old_gate[-1])
            new_gates.append(new_gate)
        else:
            # this source is being deleted
            source_map[old_index] = -1

    return np.array(new_gates, dtype=np.uint32)

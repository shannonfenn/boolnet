from BoolNet.fastrand cimport random_uniform_int
import numpy as np
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def random_move(np.ndarray[np.uint32_t, ndim=2] gates,
                np.ndarray[np.uint32_t, ndim=1] changeable, 
                np.ndarray[np.uint32_t, ndim=1] sourceable,
                unsigned int Ni):
    cdef:
        size_t g, gate, terminal, cur_source, new_source, shift
        unsigned int[:] valid_connections

    # pick a random gate to move
    gate = changeable[random_uniform_int(changeable.size)]
    # pick single random input connection to move
    terminal = random_uniform_int(2)
    # pick new source for connection randomly
    # can only come from earlier node to maintain feedforwardness
    cur_source = gates[gate][terminal]
    g = gate + Ni
    if sourceable is None:
        # decide how much to shift the input
        # (gate can only connect to previous gate or input)
        shift = random_uniform_int(gate + Ni - 1) + 1
        # Get shifted connection
        new_source = (cur_source + shift) % (gate + Ni)
    else:
        valid_connections = sourceable[(sourceable < g) & (sourceable != cur_source)]

        new_source = valid_connections[random_uniform_int(valid_connections.size)]

    return (gate, terminal, new_source)


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
    cdef:
        size_t Ng, total, i
        int g
        np.ndarray[np.uint32_t, ndim=1] result

    Ng = gates.shape[0]
    total = 0

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
    result = np.zeros(total, dtype=np.uint32)
    i = 0
    for g in range(Ni, Ni + Ng): # ignore connected inputs
        if connected[g]:
            result[i] = g - Ni
            i += 1

    return result


# def percolate_connected_gates(self):
#     ''' This detects which gates are disconnected from the
#         output and moves them to the end, returning the index
#         marking where these gates begin. '''
#     connected = self.connected_gates()

# def inputTree(self, )

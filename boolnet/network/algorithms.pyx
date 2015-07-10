from boolnet.exptools.fastrand cimport random_uniform_int
import numpy as np
cimport numpy as np
import cython


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cdef sample_mask(np.uint8_t[:] mask):
    cdef size_t i, r, random_index, total
    total = 0
    for i in range(mask.size):
        total += bool(mask[i])
    random_index = random_uniform_int(total)
    r = 0
    for i in range(mask.size):
        if mask[i]:
            if r == random_index:
                return i
            r += bool(mask[i])
    return mask.size


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def random_move(np.uint32_t[:, :] gates, np.uint8_t[:] changeable, 
                np.uint8_t[:] sourceable, size_t Ni):
    cdef:
        size_t g, gate, terminal, shift
        size_t cur_source, new_source
        np.uint8_t temp 

    # pick a random gate to move
    gate = sample_mask(changeable)
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
         # so that can undo the modification which will reflect through the view
        temp = sourceable[cur_source]
        sourceable[cur_source] = 0
        new_source = sample_mask(sourceable[ : gate + Ni])
        sourceable[cur_source] = temp

    return (gate, terminal, new_source)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def connected_sources(np.uint32_t[:, :] gates, np.uint8_t[:] connected,
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

    # find and return non-zero indices

# def percolate_connected_gates(self):
#     ''' This detects which gates are disconnected from the
#         output and moves them to the end, returning the index
#         marking where these gates begin. '''
#     connected = self.connected_gates()

# def inputTree(self, )

import numpy as np
cimport numpy as np
from libcpp.deque cimport deque
from boolnet.exptools.fastrand cimport random_uniform_int
cimport boolnet.network.algorithms as algorithms


cdef struct Move:
    size_t gate
    bint terminal
    size_t new_source


cdef class BoolNetwork:
    cdef:
        readonly size_t Ng, Ni, No
        public size_t first_unevaluated_gate
        readonly bint masked
        public bint changed
        np.uint32_t[:, :] _gates
        np.uint8_t[:] _changeable, _sourceable, _connected
        deque[Move] inverse_moves
    
    cdef _check_invariants(self)

    cdef _check_mask(self)

    cpdef connected_gates(self)

    cpdef connected_sources(self)

    cpdef _update_connected(self)

    cpdef move_to_random_neighbour(self)

    cpdef Move random_move(self) except +

    cpdef apply_move(self, Move move)

    cpdef revert_move(self)

cdef class RandomBoolNetwork(BoolNetwork):
    cdef np.uint8_t[:] _transfer_functions

    cdef _check_invariants(self)
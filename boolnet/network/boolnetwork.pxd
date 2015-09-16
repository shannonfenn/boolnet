#distutils: language = c++
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
        public np.uint32_t[:, :] gates
        public np.uint8_t[:] transfer_functions
        public np.uint8_t[:] changeable, sourceable

        readonly size_t Ng, Ni, No
        readonly bint masked

        np.uint8_t[:] connected
        deque[Move] inverse_moves
    
    cpdef clean_copy(self)
    cpdef full_copy(self)

    cpdef representation(self)
    cpdef set_representation(self, BoolNetwork gates)

    # properties
    cpdef connected_gates(self)
    cpdef connected_sources(self)
    cpdef max_node_depths(self)

    # mask modification
    cdef _check_mask(self)
    cpdef _update_connected(self)
    cpdef set_mask(self, np.uint8_t[:] sourceable, np.uint8_t[:] changeable)
    cpdef remove_mask(self)
    cpdef randomise(self)

    # Move handling
    cpdef move_to_random_neighbour(self)
    cpdef Move random_move(self) except +
    cpdef apply_move(self, Move move)
    cpdef revert_move(self)
    cpdef revert_all_moves(self)
    cpdef clear_history(self)
    cpdef history_empty(self)

    cdef _check_network_invariants(self)

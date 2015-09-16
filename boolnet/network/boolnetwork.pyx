# cython: language_level=3
# distutils: language = c++
import numpy as np
cimport numpy as np
from libcpp.deque cimport deque
from boolnet.exptools.fastrand cimport random_uniform_int
cimport boolnet.network.algorithms as algorithms


cdef class BoolNetwork:    
    def __init__(self, initial_gates, transfer_functions, size_t Ni, size_t No):
        self.transfer_functions = np.array(transfer_functions, copy=True, dtype=np.uint8)

        # copy gate array
        self.gates = np.array(initial_gates, dtype=np.uint32, copy=True)
        if self.gates.size == 0:
            raise ValueError('Empty initial gates list')

        # Store size values
        self.Ng = self.gates.shape[0]
        self.Ni = Ni
        self.No = No

        # set mask to allow all gates to be sourced and modified
        self.masked = False

        self.changeable = np.ones(self.Ng, dtype=np.uint8)      
        self.sourceable = np.ones(self.Ng + Ni, dtype=np.uint8)
        self.connected = np.zeros(self.Ng + Ni, dtype=np.uint8)

        self._check_network_invariants()
        self._update_connected()

    cpdef clean_copy(self):
        cdef BoolNetwork bn
        bn = BoolNetwork(self.gates, self.transfer_functions, self.Ni, self.No)
        return bn

    cpdef full_copy(self):
        cdef BoolNetwork bn = self.clean_copy()
        bn.changeable[:] = self.changeable
        bn.sourceable[:] = self.sourceable
        bn.connected[:] = self.connected
        bn.masked = self.masked
        bn.inverse_moves = self.inverse_moves
        return bn

    def __copy__(self):
        return self.clean_copy()

    def __str__(self):
        fmt_str = ('Ni: {} Ng: {} max node depths: {}\ngates:\n{}'
                   '\ntransfer functions:\n{}')
        return fmt_str.format(self.Ni, self.Ng, self.max_node_depths(),
                              self.gates, self.transfer_functions)

    cpdef max_node_depths(self):
        # default to stored value of No
        cdef size_t Ni, No, Ng, g
        No = self.No
        Ng = self.Ng
        Ni = self.Ni
        depths = [0] * (Ng + Ni)
        for g in range(Ng):
            source_depths = [depths[inp] for inp in self.gates[g]]
            depths[g + Ni] = max(source_depths) + 1
        # return the last No values
        return depths[-No:]

    cpdef representation(self):
        return self.clean_copy()

    cpdef set_representation(self, BoolNetwork network):
        self.gates[...] = network.gates
        self.transfer_functions[...] = network.transfer_functions
        # check invariants hold
        self._check_network_invariants()

    cpdef randomise(self):
        # this has the side effect of clearing the history since we don't
        # want to compute all the inverse moves
        cdef size_t g, i, Ni
        cdef np.uint8_t[:] sources

        Ni = self.Ni
        sources = np.array(self.sourceable, copy=True, dtype=np.uint8)
        
        for g in range(self.Ng):
            sources[g + Ni] = sources[g + Ni] or self.changeable[g]
            if self.changeable[g]:
                # valid_sources if s < gate + Ni
                # modify the connections of this gate with two random inputs
                self.gates[g, 0] = algorithms.sample_bool(sources, g + Ni)
                self.gates[g, 1] = algorithms.sample_bool(sources, g + Ni)
        self.clear_history()

    cdef _check_mask(self):
        # check for validity here rather than when attempting to
        # generate a move.
        if np.sum(self.changeable) == 0:
            raise ValueError('No changeable connections.')
        if np.sum(self.sourceable) == 0:
            raise ValueError('No changeable connections.')

        first_changeable = np.flatnonzero(np.asarray(self.changeable))[0]

        if np.sum(self.sourceable[:first_changeable + self.Ni]) < 2:
            raise ValueError(('Not enough valid connections (2 required) in: '
                              'sourceable: {} changeable: {}').format(
                np.asarray(self.sourceable), np.asarray(self.changeable)))

    cpdef set_mask(self, np.uint8_t[:] sourceable, np.uint8_t[:] changeable):
        self.changeable[:] = changeable
        self.sourceable[:] = sourceable
        self.masked = True
        self._check_mask()

    cpdef remove_mask(self):
        self.changeable[:] = 1
        self.sourceable[:] = 1
        self.masked = False

    cpdef connected_gates(self):
        self._update_connected()
        return self.connected[self.Ni:]

    cpdef connected_sources(self):
        self._update_connected()
        return self.connected

    cpdef _update_connected(self):
        algorithms.connected_sources(self.gates, self.connected, self.Ni, self.No)

    cpdef move_to_random_neighbour(self):
        self.apply_move(self.random_move())

    cpdef Move random_move(self) except +:
        ''' Returns a 3-tuple of unsigned ints (gate, terminal, new_source. '''
        # In order to ensure a useful modification is generated we make
        # sure to only modify connected gates
        cdef:
            size_t gate, terminal, shift
            size_t cur_source, new_source, Ni
            np.uint8_t temp
            Move move
        Ni = self.Ni

        # pick a random gate to move
        if self.masked:
            gate = algorithms.sample_masked_bool(self.connected_gates(), self.changeable)
        else:
            gate = algorithms.sample_bool(self.connected_gates())

        # pick single random input connection to move
        terminal = random_uniform_int(2)
        # pick new source for connection randomly
        # can only come from earlier node to maintain feedforwardness
        cur_source = self.gates[gate][terminal]

        if self.masked:
             # so that can undo the modification which will reflect through the view
            temp = self.sourceable[cur_source]
            self.sourceable[cur_source] = 0
            new_source = algorithms.sample_bool(self.sourceable, gate + Ni)
            self.sourceable[cur_source] = temp
        else:
            # decide how much to shift the input
            # (gate can only connect to previous gate or input)
            shift = random_uniform_int(gate + Ni - 1) + 1
            # Get shifted connection
            new_source = (cur_source + shift) % (gate + Ni)

        return Move(gate, terminal, new_source)

    cpdef apply_move(self, Move move):
        # expects iterable with the form (gate, terminal, new_source)
        # record the inverse move
        cdef Move inverse
        inverse.gate = move.gate
        inverse.terminal = move.terminal
        inverse.new_source = self.gates[move.gate][move.terminal]
        self.inverse_moves.push_back(inverse)

        # modify the connection
        self.gates[move.gate][move.terminal] = move.new_source

    cpdef revert_move(self):
        cdef Move inverse
        if not self.inverse_moves.empty():
            inverse = self.inverse_moves.back()
            self.inverse_moves.pop_back()
            self.gates[inverse.gate][inverse.terminal] = inverse.new_source
        else:
            raise RuntimeError('Tried to revert with empty inverse move list.')

    cpdef revert_all_moves(self):
        while not self.inverse_moves.empty():
            self.revert_move()

    cpdef clear_history(self):
        self.inverse_moves.clear()

    cpdef history_empty(self):
        return self.inverse_moves.empty()

    cdef _check_network_invariants(self):
        if self.Ng == 0:
            raise ValueError('Empty initial gates list')
        if self.Ni <= 0:
            raise ValueError('Invalid Ni ({})'.format(self.Ni))
        if self.No <= 0:
            raise ValueError('Invalid No ({})'.format(self.No))
        if self.No > self.Ng:
            raise ValueError('No > Ng ({}, {})'.format(self.No, self.Ng))
        if self.gates.ndim != 2:
            raise ValueError('initial_gates must be 2D')
        if self.gates.shape[0] != self.Ng and self.gates.shape[1] != 2:
            raise ValueError('Wrong shape ({}, {}) for gate matrix, Ng={}.'.
                format(self.gates.shape[0], self.gates.shape[1], self.Ng))
        if self.transfer_functions.ndim != 1 or self.transfer_functions.shape[0] != self.Ng:
            raise ValueError('Invalid transfer function matrix shape: {}'.format(
                self.transfer_functions.shape))
        if max(self.transfer_functions) > 15:
            raise ValueError('Invalid transfer function matrix max value: {}'.format(
                max(self.transfer_functions)))

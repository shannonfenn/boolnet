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
        readonly bint masked, changed
        np.uint32_t[:, :] _gates
        np.uint8_t[:] _changeable, _sourceable, _connected
        deque[Move] inverse_moves
    
    def __init__(self, initial_gates, size_t Ni, size_t No):
        self.changed = True
        self.first_unevaluated_gate = 0

        # copy gate array
        self._gates = np.array(initial_gates, dtype=np.uint32, copy=True)
        if self._gates.size == 0:
            raise ValueError('Empty initial gates list')

        # Store size values
        self.Ng = self._gates.shape[0]
        self.Ni = Ni
        self.No = No

        # set mask to allow all gates to be sourced and modified
        self.masked = False

        self._changeable = np.ones(self.Ng, dtype=np.uint8)      
        self._sourceable = np.ones(self.Ng + Ni, dtype=np.uint8)
        self._connected = np.zeros(self.Ng + Ni, dtype=np.uint8)

        self._check_invariants()

    cdef _check_invariants(self):
        ''' [TODO] Add more here. '''
        if self.Ng == 0:
            raise ValueError('Empty initial gates list')
        if self.Ni <= 0:
            raise ValueError('Invalid Ni ({})'.format(self.Ni))
        if self.No <= 0:
            raise ValueError('Invalid No ({})'.format(self.No))
        if self.No > self.Ng:
            raise ValueError('No > Ng ({}, {})'.format(self.No, self.Ng))
        if self._gates.ndim != 2 or self._gates.shape[1] != 2:
            raise ValueError('initial_gates must be 2D with dim2==2')
        # could have a check for recurrence

    def __str__(self):
        return ('Ni: {} Ng: {} changed: {} max node depths: {}\n'
                'gates:\n{}').format(self.Ni, self.Ng, self.changed,
                                     self.max_node_depths(), self._gates)

    property sourceable:
        def __get__(self):
            return self._sourceable

    property changeable:
        def __get__(self):
            return self._changeable

    property gates:
        def __get__(self):
            return self._gates

    def max_node_depths(self):
        # default to stored value of No
        cdef size_t Ni, No, Ng, g
        No = self.No
        Ng = self.Ng
        Ni = self.Ni
        depths = [0] * (Ng + Ni)
        for g in range(Ng):
            depths[g + Ni] = max(depths[inp] for inp in self._gates[g]) + 1
        # return the last No values
        return depths[-No:]

    def force_reevaluation(self):
        self.changed = True

    def reconnect_masked_range(self):
        # this has the side effect of clearing the history since we don't
        # want to compute all the inverse moves
        cdef size_t g, i, Ni
        cdef np.uint8_t[:] sources

        Ni = self.Ni
        sources = np.array(self._sourceable, copy=True, dtype=np.uint8)

        self.clear_history()
        
        for g in range(self.Ng):
            sources[g + Ni] = sources[g + Ni] or self._changeable[g]
            if self.changeable[g]:
                # valid_sources if s < gate + Ni
                # modify the connections of this gate with two random inputs
                self._gates[g, 0] = algorithms.sample_mask(sources, g + Ni)
                self._gates[g, 1] = algorithms.sample_mask(sources, g + Ni)

        if self.changed:
            self.first_unevaluated_gate = min(self.first_unevaluated_gate,
                                              min(self._changeable))
        else:
            self.first_unevaluated_gate = min(self._changeable)
        # indicate the network must be reevaluated
        self.changed = True

    cdef _check_mask(self):
        # check for validity here rather than when attempting to
        # generate a move.
        if np.sum(self._changeable) == 0:
            raise ValueError('No changeable connections.')
        if np.sum(self._sourceable) == 0:
            raise ValueError('No changeable connections.')

        first_changeable = np.flatnonzero(self._changeable)[0]

        if self._sourceable[:first_changeable + self.Ni].sum() < 2:
            raise ValueError(('Not enough valid connections (2 required) in: '
                              'sourceable: {} changeable: {}').format(
                self._sourceable, self._changeable))

    def set_mask(self, np.uint8_t[:] sourceable, np.uint8_t[:] changeable):
        self._changeable[:] = changeable
        self._sourceable[:] = sourceable
        self.masked = True
        self._check_mask()

    def remove_mask(self):
        # don't need to bother removing the actual masks
        self._changeable[:] = 1
        self._sourceable[:] = 1
        self.masked = False

    cpdef Move random_move(self):
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
            gate = algorithms.sample_mask(self._changeable & self.connected_gates())
        else:
            gate = algorithms.sample_mask(self.connected_gates())

        # pick single random input connection to move
        terminal = random_uniform_int(2)
        # pick new source for connection randomly
        # can only come from earlier node to maintain feedforwardness
        cur_source = self._gates[gate][terminal]

        if self.masked:
             # so that can undo the modification which will reflect through the view
            temp = self._sourceable[cur_source]
            self._sourceable[cur_source] = 0
            new_source = algorithms.sample_mask(self._sourceable, gate + Ni)
            self._sourceable[cur_source] = temp
        else:
            # decide how much to shift the input
            # (gate can only connect to previous gate or input)
            shift = random_uniform_int(gate + Ni - 1) + 1
            # Get shifted connection
            new_source = (cur_source + shift) % (gate + Ni)

        return Move(gate, terminal, new_source)

    cpdef connected_gates(self):
        self._update_connected()
        return self._connected[self.Ni:]

    cpdef connected_sources(self):
        self._update_connected()
        return self._connected

    cpdef _update_connected(self):
        algorithms.connected_sources(self._gates, self._connected, self.Ni, self.No)

    cpdef move_to_random_neighbour(self):
        self.move_to_given(self.random_move())

    cpdef move_to_given(self, Move move):
        # expects iterable with the form (gate, terminal, new_source)
        if self.changed:
            self.first_unevaluated_gate = min(self.first_unevaluated_gate, move.gate)
        else:
            self.first_unevaluated_gate = move.gate

        # record the inverse move
        inverse = (move.gate, move.terminal, self._gates[move.gate][move.terminal])
        self._inverse_moves.append(inverse)

        # modify the connection
        self._gates[move.gate][move.terminal] = move.new_source
        # indicate the network must be reevaluated
        self.changed = True

    def revert_move(self):
        try:
            inverse = self._inverse_moves.pop()
            if self.changed:
                # if multiple moves are undone there are no issues with
                # recomputation since the earliest gate ever changed will
                # be the startpoint
                self.first_unevaluated_gate = min(self.first_unevaluated_gate, inverse.gate)
            else:
                self.first_unevaluated_gate = inverse.gate
            self.changed = True
            self._gates[inverse.gate][inverse.terminal] = inverse.new_source

        except IndexError:
            raise RuntimeError('Tried to revert with empty inverse move list.')

    def revert_all_moves(self):
        while self._inverse_moves:
            self.revert_move()

    def clear_history(self):
        self._inverse_moves.clear()


cdef class RandomBoolNetwork(BoolNetwork):
    cdef np.uint8_t[:] _transfer_functions

    def __init__(self, initial_gates, Ni, No, transfer_functions):
        self._transfer_functions[:] = transfer_functions
        super().__init__(initial_gates, Ni, No)

    cdef _check_invariants(self):
        super()._check_invariants()
        if self._transfer_functions.shape != (self.Ng,):
            raise ValueError('Invalid transfer function matrix shape: {}'.format(
                self._transfer_functions.shape))

    def __str__(self):
        return super().__str__() + '\ntransfer functions:\n{}'.format(self._transfer_functions)

    property transfer_functions:
        def __get__(self):
            return self._transfer_functions

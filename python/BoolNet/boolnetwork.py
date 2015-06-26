import numpy as np
try:
    import BoolNet.NetworkAlgorithms as NetworkAlgorithms
except ImportError:
    import pyximport
    pyximport.install()
    import BoolNet.NetworkAlgorithms as NetworkAlgorithms


class BoolNetwork:

    def __init__(self, initial_gates, Ni, No):
        self._evaluated = False
        self.first_unevaluated_gate = 0
        self._inverse_moves = []

        # copy gate array
        self._gates = np.array(initial_gates, dtype=np.uint32, copy=True)
        if self._gates.size == 0:
            raise ValueError('Empty initial gates list')

        # Store size values
        self.Ng = self._gates.shape[0]
        self.Ni = Ni
        self.No = No

        # set mask to allow all gates to be sourced and modified
        self._masked = False
        self._changeable = np.array([], dtype=np.uint32)
        self._sourceable = np.array([], dtype=np.uint32)

        self._connected = np.zeros(self.Ng + self.Ni, dtype=np.uint8)

        self._check_invariants()

    def _check_invariants(self):
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
        return ('Ni: {} Ng: {} evaluated: {} max node depths: {}\n'
                'gates:\n{}').format(self.Ni, self.Ng, self._evaluated,
                                     self.max_node_depths(), self._gates)

    @property
    def masked(self):
        return self._masked

    @property
    def sourceable(self):
        return self._sourceable if self._masked else np.array([], dtype=np.uint32)

    @property
    def changeable(self):
        return self._changeable if self._masked else np.array([], dtype=np.uint32)

    @property
    def gates(self):
        return self._gates

    def max_node_depths(self, No=None):
        # default to stored value of No
        if No is None:
            No = self.No
        Ng = self.Ng
        Ni = self.Ni
        depths = [0] * (Ng + Ni)
        for g in range(Ng):
            depths[g + Ni] = max(depths[inp] for inp in self._gates[g]) + 1
        # return the last No values
        return depths[-No:]

    def force_reevaluation(self):
        self._evaluated = False

    def reconnect_masked_range(self):
        # this has the side effect of clearing the history since we don't
        # want to compute all the inverse moves
        self.clear_history()

        # sources = self._sourceable | set(c + self.Ni for c in self._changeable)
        sources = np.union1d(self._sourceable, self._changeable + self.Ni)
        for gate in self._changeable:
            # valid_sources = [s for s in sources if s < gate + self.Ni]
            valid_sources = sources[sources < (gate + self.Ni)]
            # modify the connections of this gate with two random inputs
            self._gates[gate] = np.random.choice(valid_sources, size=2)

        if self._evaluated:
            self.first_unevaluated_gate = min(self._changeable)
        else:
            self.first_unevaluated_gate = min(self.first_unevaluated_gate,
                                              min(self._changeable))
        # indicate the network must be reevaluated
        self._evaluated = False

    def _check_mask(self):
        # check for validity here rather than when attempting to
        # generate a move.
        pure_inputs = np.setdiff1d(self._sourceable, self._changeable + self.Ni)

        if np.sum(pure_inputs < np.amin(self._changeable) + self.Ni) < 2:
            raise ValueError(('Not enough valid connections (2 required) in: '
                              'sourceable: {} changeable: {} pure inputs {}').format(
                self._sourceable, self._changeable, pure_inputs))

    def set_mask(self, sourceable, changeable):
        self._changeable = np.unique(np.asarray(changeable, dtype=np.uint32))
        self._sourceable = np.unique(np.asarray(sourceable, dtype=np.uint32))
        self._masked = True
        self._check_mask()

    def remove_mask(self):
        # don't need to bother removing the actual masks
        self._masked = False

    def random_move(self):
        ''' Returns a 3-tuple of unsigned ints (gate, terminal, new_source. '''
        # In order to ensure a useful modification is generated we make
        # sure to only modify connected gates

        # the result of connected_gates() includeds inputs and is Ni-based
        # so needs to be shifted and normalised
        changeable = self.connected_gates()

        if self._masked:
            # this will cause problems if the mask has been set poorly and there is
            # no intersection between connected and changeable
            changeable = np.intersect1d(self._changeable, changeable, assume_unique=True)
            sourceable = self._sourceable
        else:
            sourceable = None

        return NetworkAlgorithms.random_move(self._gates, changeable, sourceable, self.Ni)

    def connected_gates(self, No=None):
        ''' This detects which gates and inputs are connected to the output
            and returns the gate indices as a sorted list. This
            is just the union of the connected components in the
            digraph where connections are only backward.'''
        if No is None:
            No = self.No

        return NetworkAlgorithms.connected_sources(
            self._gates, self._connected, self.Ni, No)

    # def percolate_connected_gates(self):
    #     ''' This detects which gates are disconnected from the
    #         output and moves them to the end, returning the index
    #         marking where these gates begin. '''
    #     connected = self.connected_gates()

    # def inputTree(self, )

    def move_to_neighbour(self, move):
        # expects iterable with the form (gate, terminal, new_source)
        if self._evaluated:
            self.first_unevaluated_gate = move[0]
        else:
            self.first_unevaluated_gate = min(self.first_unevaluated_gate, move[0])

        # record the inverse move
        inverse = (move[0], move[1], self._gates[move[0]][move[1]])
        self._inverse_moves.append(inverse)

        # modify the connection
        self._gates[move[0]][move[1]] = move[2]
        # indicate the network must be reevaluated
        self._evaluated = False

    def revert_move(self):
        try:
            inverse = self._inverse_moves.pop()
            if self._evaluated:
                self.first_unevaluated_gate = inverse[0]
            else:
                # if multiple moves are undone there are no issues with
                # recomputation since the earliest gate ever changed will
                # be the startpoint
                self.first_unevaluated_gate = min(self.first_unevaluated_gate, inverse[0])
            self._evaluated = False
            self._gates[inverse[0]][inverse[1]] = inverse[2]

        except IndexError:
            raise RuntimeError('Tried to revert with empty inverse move list.')

    def revert_all_moves(self):
        while self._inverse_moves:
            self.revert_move()

    def clear_history(self):
        self._inverse_moves = []

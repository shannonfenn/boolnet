# distutils: language = c++
# distutils: sources = BitError.cpp BooleanNetworkNAND.cpp
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args = -std=c++11

################# REMOVE FOR FASTER CODE ######################

from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libcpp cimport bool
from cython.operator cimport dereference as deref
from random import randint, getrandbits, choice
from copy import copy

cdef extern from "BitError.hpp" namespace "BitError":
    cpdef enum Metric:
        SIMPLE,
        WEIGHTED_LIN_MSB,        WEIGHTED_LIN_LSB,
        WEIGHTED_EXP_MSB,        WEIGHTED_EXP_LSB,
        HIERARCHICAL_LIN_MSB,    HIERARCHICAL_LIN_LSB,
        HIERARCHICAL_EXP_MSB,    HIERARCHICAL_EXP_LSB,
        WORST_SAMPLE_LIN_MSB,    WORST_SAMPLE_LIN_LSB,
        WORST_SAMPLE_EXP_MSB,    WORST_SAMPLE_EXP_LSB,
        E4_MSB,                  E4_LSB,
        E5_MSB,                  E5_LSB,
        E6_MSB,                  E6_LSB,
        E7_MSB,                  E7_LSB

__METRIC_NAME_MAP = {
     'simple'                       : SIMPLE,
     'weighted linear msb'          : WEIGHTED_LIN_MSB,        
     'weighted linear lsb'          : WEIGHTED_LIN_LSB,        
     'weighted exponential msb'     : WEIGHTED_EXP_MSB,        
     'weighted exponential lsb'     : WEIGHTED_EXP_LSB,        
     'hierarchical linear msb'      : HIERARCHICAL_LIN_MSB,    
     'hierarchical linear lsb'      : HIERARCHICAL_LIN_LSB,        
     'hierarchical exponential msb' : HIERARCHICAL_EXP_MSB,    
     'hierarchical exponential lsb' : HIERARCHICAL_EXP_LSB,        
     'worst sample linear msb'      : WORST_SAMPLE_LIN_MSB,    
     'worst sample linear lsb'      : WORST_SAMPLE_LIN_LSB,        
     'worst sample exponential msb' : WORST_SAMPLE_EXP_MSB,    
     'worst sample exponential lsb' : WORST_SAMPLE_EXP_LSB,
     'e4 msb'                       : E4_MSB,
     'e4 lsb'                       : E4_LSB,
     'e5 msb'                       : E5_MSB,
     'e5 lsb'                       : E5_LSB,
     'e6 msb'                       : E6_MSB,
     'e6 lsb'                       : E6_LSB,
     'e7 msb'                       : E7_MSB,
     'e7 lsb'                       : E7_LSB}

__INV_METRIC_NAME_MAP = { v: k for k, v in __METRIC_NAME_MAP.items() }

def all_metrics():
    for m in __INV_METRIC_NAME_MAP:
        yield m

def metric_from_name(name):
    return __METRIC_NAME_MAP[name]

def metric_name(metric):
    return __INV_METRIC_NAME_MAP[metric]

cdef extern from "BooleanNetworkNAND.h":
    cdef struct Move:
        size_t gate
        size_t connection
        bool inp # false means first

    cdef cppclass BooleanNetworkNAND:

        BooleanNetworkNAND(vector[vector[size_t]] initial_gates, 
                           vector[vector[char]] inputs, 
                           vector[vector[char]] target) except +

        BooleanNetworkNAND(BooleanNetworkNAND& old)

        size_t getNi()
        size_t getNo()
        size_t getNg()

        # void addGates(size_t N)
        vector[vector[size_t]] getGates()

        double accuracy()
        double error(Metric metric)
        vector[double] errorPerBit()
        vector[vector[char]] errorMatrix()

        vector[vector[char] ] getFullStateTableForSamples()

        vector[size_t] maxDepthPerBit()

        void setExamples(vector[vector[char]] inputs, 
                         vector[vector[char]] target)  except +
        # void pertubExamples(default_random_engine& generator)

        vector[vector[char]] getTruthTable()

        # Move getRandomMove(default_random_engine& generator)
        void moveToNeighbour(const Move& move)
        void revertMove() except +
        void revertAllMoves()
        void clearHistory()

        void forceReevaluation()

        # private variables (hopefully not exposed in python side)
        size_t Ni
        size_t No
        bool evaluated
        size_t first_changed_gate
        deque[Move] inverse_moves
        vector[vector[size_t]] gates
        vector[vector[char]] state_matrix
        vector[vector[char]] target_matrix
        vector[vector[char]] error_matrix
# cdef struct Move:
#     size_t gate
#     size_t connection
#     bool inp # false means first

cdef class pyBooleanNetworkNAND:
    cdef BooleanNetworkNAND* net      # holds the C++ object this is wrapping
    cdef list _inputs
    cdef list _target
    cdef bool _masked
    cdef tuple _changeable
    cdef set _sourceable

    def __cinit__(self, vector[vector[size_t]] initial_gates, 
                        vector[vector[char]] inputs, 
                        vector[vector[char]] target):
        # allocate c++ object
        self.net = new BooleanNetworkNAND(initial_gates, inputs, target)
        # store for later access
        self._inputs = inputs
        self._target = target
        # set mask to allow all gates to be sourced and modified
        self._masked = False        
        self._changeable = tuple() # conversion to tuple to make move selection faster later
        self._sourceable = set()
        
    def __dealloc__(self):
        del self.net

    def __str__(self):
        return '''Ni: {}No: {} Ng: {} evaluated: {} first_changed_gate: {} accuracy: {} 
        error (simple): {} error per output: {} max depth per output: {} inputs: {} 
        target: {}\ngates: {}\nstate: {}\nerror matrix: {}\ntarget matrix: {}'''.format(
                    self.Ni, self.No, self.Ng, self.net.evaluated,
                    self.net.first_changed_gate, self.net.accuracy(), self.net.error(SIMPLE),
                    self.error_per_output(), self.max_depth_per_output(), self._inputs, 
                    self._target, self.net.gates, self.net.state_matrix, 
                    self.net.error_matrix, self.net.target_matrix)

    def __copy__(self):
        cdef pyBooleanNetworkNAND cp = pyBooleanNetworkNAND(self.net.gates, self._inputs, self._target)
        cp.net.evaluated = self.net.evaluated
        cp.net.first_changed_gate = self.net.first_changed_gate
        cp.net.inverse_moves = self.net.inverse_moves
        cp.net.state_matrix = self.net.state_matrix
        cp.net.target_matrix = self.net.target_matrix
        cp.net.error_matrix = self.net.error_matrix
        cp._inputs = self._inputs
        cp._target = self._target
        cp._masked = self._masked
        cp._sourceable = self._sourceable
        cp._changeable = self._changeable
        return cp
    
    property Ni:
        def __get__(self):
            return self.net.Ni
    property No:
        def __get__(self):
            return self.net.No
    property Ng:
        def __get__(self):
            return self.net.getNg()

    # def add_gates(self, N):
    #    self.net.addGates(N)

    cpdef gates(self):
       return self.net.getGates()

    def accuracy(self):
       return self.net.accuracy()
    def error(self, Metric metric):
       return self.net.error(metric)
    def error_per_output(self):
       return self.net.errorPerBit()
    def error_matrix(self):
       return self.net.errorMatrix()

    def full_state_table_for_samples(self):
        return self.net.getFullStateTableForSamples()

    def max_depth_per_output(self):
        return self.net.maxDepthPerBit()
    
    def set_examples(self, inputs, target):
        self.net.setExamples(inputs, target)

    # void pertubExamples(default_random_engine& generator):
    # 	return self.net.getNi()
    
    def truth_table(self):
        return self.net.getTruthTable()

    property inputs:
        def __get__(self):
            return self._inputs

    property target:
        def __get__(self):
            return self._target

    property masked:
        def __get__(self):
            return self._masked

    property sourceable:
        def __get__(self):
            return self._sourceable if self._masked else set()

    property changeable:
        def __get__(self):
            return self._changeable if self._masked else set()

    def set_mask(self, changeable, sourceable):
        # some basic checks for validity (done here than when attempting to generate a move).
        if not changeable or not sourceable:
            raise ValueError('''Empty changeable or sourceable: 
                            \n\tsourceable: {}\n\tchangeable: {}
                            '''.format(sourceable, changeable))
        # more in depth validity checks
        earliest_changeable = min(changeable)
        g = earliest_changeable + self.net.Ni
        valid_connections = { c for c in sourceable if c < g }

        # in reality this would only happen if the kFS stage found a 1 feature set, 
        # which in reality is already learnt and should be caught earlier
        if len(valid_connections) < 2:
            raise ValueError('''Less than two valid connections from: 
                            \nsourceable: {}\nchangeable: {}\n
                            gate: {}'''.format(sourceable, changeable, earliest_changeable))
        # sets are valid, copy them in
        # conversion to tuple to make move selection faster later
        self._changeable = tuple(set(changeable)) 
        self._sourceable = set(sourceable)

        self._masked = True

    def remove_mask(self):
        # don't need to bother removing the actual masks
        self._masked = False

    def force_reevaluation(self):
        self.net.forceReevaluation()

    # def restricted_random_move(self, sourceable, changeable):
    #     ''' This provides random move, only selecting gates to change
    #         from "changeable" and only selecting new inputs
    #         from "sourceable". Both should be sets. '''
    #     cdef Move move

    #     # pick a gate to move
    #     try:
    #         move.gate = choice(tuple(changeable))
    #     except IndexError:
    #         raise ValueError('''Empty changeable or sourceable: 
    #                         \n\tsourceable: {}\n\tchangeable: {}
    #                         '''.format(sourceable, changeable))

    #     # pick single random input connection to move
    #     move.inp = getrandbits(1)
    #     # pick new source for connection randomly
    #     # can only come from earlier node to maintain feedforwardness

    #     prev_inps = self.net.gates[move.gate]
    #     prev_con = prev_inps[move.inp]
    #     g = move.gate + self.net.Ni
    #     valid_connections = tuple( c for c in sourceable if c < g and c != prev_con )
        
    #     try:
    #         move.connection = choice(valid_connections)
    #     except IndexError:
    #         raise ValueError('''Empty valid connections from: 
    #                         \nsourceable: {}\nchangeable: {}\n
    #                         gate: {} previous connection: {} input: {}'''.format(
    #                         sourceable, changeable, prev_con, move.gate, move.inp))

    #     return move 

    def random_move(self):
        cdef Move move
        if self._masked:
            # pick a gate to move
            move.gate = choice(self._changeable)

            # pick single random input connection to move
            move.inp = getrandbits(1)
            # pick new source for connection randomly
            # can only come from earlier node to maintain feedforwardness

            prev_inps = self.net.gates[move.gate]
            prev_con = prev_inps[move.inp]
            g = move.gate + self.net.Ni
            valid_connections = tuple( c for c in self._sourceable if c < g and c != prev_con )
            
            move.connection = choice(valid_connections)

        else:
            # pick a gate to move
            move.gate = randint(0, self.net.getNg() - 1)
            # pick single random input connection to move
            move.inp = getrandbits(1)
            # decide how much to shift the input 
            # (gate can only connect to previous gate or input)
            shift = randint(1, move.gate + self.net.Ni - 1)
            # Get location of the original connection
            previous_connection = self.gates()[move.gate][move.inp]
            # Get shifted connection
            move.connection = (previous_connection + shift) % (move.gate + self.net.Ni)
            
        return move 

    # def percolate(self):
    #     ''' This detects which gates are disconnected from the output
    #         and moves them to the end, returning the index marking where
    #         these gates begin. '''
    #     # first determine which gates are connected to the output, this is 
    #     # just the union of the connected components in the digraph where
    #     # connections are only backward
    #     connected = set()
    #     for o in xrange(No()):


    # def inputTree(self, )

    def move_to_neighbour(self, Move move):
        self.net.moveToNeighbour(move)
    def revert_move(self):
        self.net.revertMove()
    def revert_all_moves(self):
        self.net.revertAllMoves()
    def clear_history(self):
        self.net.clearHistory()

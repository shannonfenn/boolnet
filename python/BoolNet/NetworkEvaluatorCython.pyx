from BoolNet.NetworkEvaluator import NetworkEvaluator
import BoolNet.BitErrorCython as BitErrorCython
import numpy as np
cimport numpy as np
from libc.math cimport log2, ceil
import cython
from copy import deepcopy

from BoolNet.Packing cimport packed_type_t, PACKED_SIZE, generate_end_mask, f_type, function_list
from BoolNet.Packing import packed_type

# cdef struct State:
#     BooleanNetwork network
#     np.uint8[:, :] activation
#     np.uint8[:, :] output
#     np.uint8[:, :] error


def _check_combined_invariants(network, 
                               packed_type_t[:,:] inputs,
                               packed_type_t[:,:] target):
    ''' [TODO] Add more here. '''
    if network.No != target.shape[0]:
        raise ValueError(('Network output # ({}) does not matchexpress target '
                         '({}).').format(network.No, target.shape[0]))


def _check_instance_invariants(packed_type_t[:,:] inputs, 
                               packed_type_t[:,:] target,
                               unsigned int Ne):
    Ni = inputs.shape[0]
    No = target.shape[0]
    # Test if the matrices are valid
    if inputs.shape[1] == 0 or Ni == 0:
        raise ValueError('Empty input matrix.')
    if target.shape[1] == 0 or No == 0:
        raise ValueError('Empty target matrix.')
    if 2**Ni < Ne:
        raise ValueError('More examples ({}) than #inputs ({}) '
                         'can represent.'.format(Ne, Ni))
    if inputs.shape[1] != target.shape[1]:
        raise ValueError('Incompatible input/target shapes: {} {}.'.format(
                         inputs.shape[1], target.shape[1]))


cdef class NetworkEvaluatorCython:
    cdef readonly unsigned int Ne
    cdef packed_type_t[:, :] inputs, target
    cdef packed_type_t zero_mask
    cdef list states    # network, activation, output, error, error_scratch

    def __init__(self,
                 packed_type_t[:,:] inputs,
                 packed_type_t[:,:] target,
                 unsigned int Ne):
        # check invariants hold
        _check_instance_invariants(inputs, target, Ne)
        self.Ne = Ne
        # transpose and pack into integers
        self.inputs = np.array(inputs)
        self.target = np.array(target)

        self.zero_mask = generate_end_mask(Ne)

        self.states = []

    property input_matrix:
        def __get__(self):
            return self.inputs

    property target_matrix:
        def __get__(self):
            return self.target

    property num_networks:
        def __get__(self):
            return len(self.states)

    def network(self, unsigned int index):
        return self.states[index][0]

    def activation_matrix(self, unsigned int index):
        self.evaluate(index)
        return self.states[index][1]

    def output_matrix(self, unsigned int index):
        self.evaluate(index)
        return self.states[index][2]

    def error_matrix(self, unsigned int index):
        self.evaluate(index)
        return self.states[index][3]

    def add_network(self, network):
        ''' Set up the activation and error matrices for a new network and add the
            resulting State struct to the states list.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        _check_combined_invariants(network, self.inputs, self.target)

        cdef:
            packed_type_t[:, :] activation, error, error_scratch
            size_t Ni, No, Ne

        Ni, Ne = self.inputs.shape[0], self.inputs.shape[1]
        No = self.target.shape[0]
        # instantiate a matrix for activation and for error
        activation = np.empty((network.Ng + Ni, Ne), dtype=packed_type)
        error = np.empty_like(self.target)
        error_scratch = np.empty_like(self.target)

        # copy inputs into activation matrix
        activation[:Ni, :] = self.inputs

        # this prevents another evaluator causing problems with this network
        network = deepcopy(network)
        # indicate that the network needs to be evaluated later
        network._evaluated = False
        network.first_unevaluated_gate = 0

        # append evaluation struct to list of states
        self.states.append((network, activation, activation[-No:, :], error, error_scratch))
        return len(self.states) - 1

    def remove_network(self, unsigned int index):
        self.states.pop(index)

    def remove_all_networks(self):
        self.states = []

    def metric_value(self, size_t index, metric):
        self.evaluate(index)
        return BitErrorCython.metric_value(
            self.states[index][3], self.states[index][4], self.Ne, self.zero_mask, metric)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _zero_mask_last_column(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the
        # last column in the error matrix with to clear the value back
        # to zero
        cdef:
            unsigned int r
            unsigned int rows = matrix.shape[0]
            unsigned int cols = matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask

    cpdef evaluate(self, unsigned int index):
        ''' Evaluate the activation and error matrices for the selected network if it
            has been modified since the last evaluation. '''
        state = self.states[index]

        if state[0]._evaluated:
            return

        if hasattr(state[0], 'transfer_functions'):
            self._evaluate_random(state)
        else:
            self._evaluate_NAND(state)

        self._zero_mask_last_column(state[1])   # this does 2 as well (since it is a view)
        self._zero_mask_last_column(state[3])

        state[0]._evaluated = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _evaluate_random(self, state):
        ''' Evaluate the activation and error matrices for the selected network
            getting node TFs from network. '''
        cdef:
            size_t Ni, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:,:] activation, output, error, target
            np.uint8_t[:] transfer_functions
            np.uint32_t[:, :] gates
            f_type func

        Ng = state[0].Ng
        No = self.target.shape[0]
        Ni = self.inputs.shape[0]
        network = state[0]
        activation = state[1]
        output = state[2]
        error = state[3]
        target = self.target
        gates = network.gates
        transfer_functions = network.transfer_functions

        cols = activation.shape[1]

        # evaluate the state matrix
        start = network.first_unevaluated_gate

        for g in range(start, Ng):
            in1 = gates[g, 0]
            in2 = gates[g, 1]
            func = function_list[transfer_functions[g]]
            for c in range(cols):
                activation[Ni+g, c] = func(activation[in1, c], activation[in2, c])
        # evaluate the error matrix

        for o in range(No):
            for c in range(cols):
                error[o, c] = (target[o, c] ^ output[o, c])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _evaluate_NAND(self, state):
        ''' Evaluate the activation and error matrices for the selected network
            assuming each node TF is NAND. '''
        cdef:
            size_t Ni, No, Ng, Ne, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, output, error, target
            np.uint32_t[:, :] gates
        
        Ng = state[0].Ng
        Ni = self.inputs.shape[0]
        No = self.target.shape[0]
        network = state[0]
        activation = state[1]
        output = state[2]
        error = state[3]
        target = self.target
        gates = network.gates

        cols = activation.shape[1]

        # evaluate the state matrix
        start = network.first_unevaluated_gate
        for g in range(start, Ng):
            in1 = gates[g, 0]
            in2 = gates[g, 1]
            for c in range(cols):
                activation[Ni+g, c] = ~(activation[in1, c] & activation[in2, c])
        # evaluate the error matrix
        for o in range(No):
            for c in range(cols):
                error[o, c] = (target[o, c] ^ output[o, c])







    # def truth_table(self, index):
    #     ''' Generate and return the full truth table for the chosen network.
    #         WARNING: This is exponentially large with Ni. '''
    #     state = self._states[index]

    #     if hasattr(state.network, 'transfer_functions'):
    #         return self._tt_random(state)
    #     else:
    #         return self._tt_NAND(state)

    # def _tt_random(self, state):
    #     ''' Generate and return the full truth table for the chosen network
    #         getting node TFs from network.
    #         WARNING: This is exponentially large with Ni. '''
    #     Ni = self.inputs.shape[1]
    #     No = self.target.shape[1]
    #     gates = state.network.gates
    #     TF = state.network.transfer_functions

    #     activation = np.zeros(Ni + state.network.Ng, dtype=np.uint8)
    #     output_table = np.empty((2**Ni, No), dtype=np.uint8)

    #     for i in range(2**Ni):
    #         # Generate next input
    #         activation[:Ni] = [(1 << b & i) != 0 for b in range(Ni)]
    #         # evaluate state vector
    #         for g, gate in enumerate(gates):
    #             activation[Ni + g] = TF[g, activation[gate[0]], activation[gate[1]]]
    #         # copy output states to table
    #         output_table[i, :] = activation[-No:]

    #     return output_table

    # def _tt_NAND(self, state):
    #     ''' Generate and return the full truth table for the chosen network
    #         assuming each node TF is NAND.
    #         WARNING: This is exponentially large with Ni. '''
    #     Ni = self.inputs.shape[1]
    #     No = self.target.shape[1]
    #     gates = state.network.gates

    #     activation = np.zeros(Ni + state.network.Ng, dtype=np.uint8)
    #     output_table = np.empty((2**Ni, No), dtype=np.uint8)

    #     for i in range(2**Ni):
    #         # Generate next input
    #         activation[:Ni] = [(1 << b & i) != 0 for b in range(Ni)]
    #         # evaluate state vector
    #         for g, gate in enumerate(gates):
    #             activation[Ni + g] = not(activation[gate[0]] and
    #                                      activation[gate[1]])
    #         # copy output states to table
    #         output_table[i, :] = activation[-No:]

    #     return output_table

import cython
import numpy as np
cimport numpy as np
from copy import deepcopy

from BoolNet.BooleanNetwork import BooleanNetwork
import BoolNet.BitErrorCython as BitError
from BoolNet.Packing cimport packed_type_t, generate_end_mask, f_type, function_list


cdef _check_invariants(network,
                       packed_type_t[:, :] inputs,
                       packed_type_t[:, :] target,
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
    # [TODO] Add more here for combined invariants.
    if network.No != target.shape[0]:
        raise ValueError(('Network output # ({}) does not matchexpress target '
                         '({}).').format(network.No, target.shape[0]))


cdef class NetworkState:
    cdef:
        readonly unsigned int Ne, Ni, No, Ng
        packed_type_t[:, :] activation, inputs, output,
                            target, error, error_scratch
        packed_type_t zero_mask
        BooleanNetwork network

    def __init__(self,
                 packed_type_t[:, :] inputs,
                 packed_type_t[:, :] target,
                 BooleanNetwork network,
                 unsigned int Ne):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        # check invariants hold
        _check_invariants(network, inputs, target, Ne)

        self.Ne = Ne
        self.Ni = inputs.shape[0]
        self.No = target.shape[0]

        # transpose and pack into integers
        self.target = np.array(target)

        self.zero_mask = generate_end_mask(Ne)
        
        # instantiate a matrix for activation
        self.activation = np.empty((network.Ng + self.Ni, inputs.shape[1]), dtype=packed_type)
        # copy inputs into activation matrix
        self.activation[:self.Ni, :] = self.inputs
        # create output view into activation matrix
        self.output = self.activation[-self.No:, :]

        # instantiate matrices for error
        error = np.empty_like(self.target)
        error_scratch = np.empty_like(self.target)

        # prevent another evaluator causing problems with this network
        self.network = deepcopy(network)
        # force reevaluation of the copied network
        self.network._evaluated = False
        self.network.first_unevaluated_gate = 0

    property input_matrix:
        def __get__(self):
            return self.inputs

    property target_matrix:
        def __get__(self):
            return self.target

    property activation_matrix:
        def __get__(self):
            self.evaluate()
            return self.activation

    property output_matrix:
        def __get__(self):
            self.evaluate()
            return self.output_matrix

    property error_matrix:
        def __get__(self):
            self.evaluate()
            return self.error_matrix

    def metric_value(self, metric):
        self.evaluate()
        return BitError.metric_value(self.error_matrix, self.error_scratch,
                                           self.Ne, self.zero_mask, metric)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _zero_mask_last_column(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef:
            unsigned int r
            unsigned int rows = matrix.shape[0]
            unsigned int cols = matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask

    cpdef evaluate(self):
        ''' Evaluate the activation and error matrices if the
            network has been modified since the last evaluation. '''

        if self.network._evaluated:
            return

        if hasattr(self.network, 'transfer_functions'):
            self._evaluate_random()
        else:
            self._evaluate_NAND()

        self._zero_mask_last_column(self.activation)   # this does output_matrix as well (since it is a view)
        self._zero_mask_last_column(self.error)

        self.network._evaluated = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _evaluate_random(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, output, error, target
            np.uint8_t[:] transfer_functions
            np.uint32_t[:, :] gates
            f_type func

        Ng = self.Ng
        Ni = self.Ni
        No = self.No
        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        output = self.output
        error = self.error
        target = self.target
        gates = self.network.gates
        transfer_functions = self.network.transfer_functions

        cols = activation.shape[1]

        # evaluate the state matrix
        start = self.network.first_unevaluated_gate

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
    cdef void _evaluate_NAND(self):
        ''' Evaluate the activation and error matrices for the network
            assuming each node TF is NAND. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, output, error, target
            np.uint32_t[:, :] gates
                
        Ng = self.Ng
        Ni = self.Ni
        No = self.No

        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        output = self.output
        error = self.error
        target = self.target
        gates = self.network.gates
        
        cols = activation.shape[1]

        # evaluate the state matrix
        start = self.network.first_unevaluated_gate
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

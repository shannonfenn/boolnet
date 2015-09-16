# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
from copy import copy

from boolnet.network.boolnetwork cimport BoolNetwork, Move
from boolnet.bintools.functions cimport Function
from boolnet.bintools.biterror import STANDARD_EVALUATORS
from boolnet.bintools.biterror cimport StandardEvaluator
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.biterror_chained cimport ChainedEvaluator
from boolnet.bintools.packing cimport packed_type_t, generate_end_mask, f_type, function_list, PACKED_SIZE
from boolnet.bintools.packing import packed_type
from boolnet.bintools.example_generator cimport PackedExampleGenerator, OperatorExampleIteratorFactory


# cpdef standard_from_mapping(network, mapping):
#     if isinstance(mapping, FileBoolMapping):
#         return StandardNetworkState(network, mapping.inputs, mapping.target, mapping.Ne)
#     elif isinstance(mapping, OperatorBoolMapping):
#         return standard_from_operator(network, mapping.indices,
#                                       mapping.Nb, mapping.No,
#                                       mapping.operator, mapping.N)


cpdef standard_from_operator(gates, indices, Nb, No, operator, N=0):
    cdef packed_type_t[:, :] inp, tgt
    ex_factory = OperatorExampleIteratorFactory(indices, Nb, operator, N)
    packed_factory = PackedExampleGenerator(ex_factory, No)

    Ni = packed_factory.Ni
    Ne = packed_factory.Ne

    chunks = Ne // PACKED_SIZE
    if packed_factory.Ne % PACKED_SIZE > 0:
        chunks += 1

    inp = np.empty((Ni, chunks), dtype=packed_type)
    tgt = np.empty((No, chunks), dtype=packed_type)

    packed_factory.reset()
    packed_factory.next_examples(inp, tgt)
    return StandardNetworkState(gates, inp, tgt, Ne)


cpdef chained_from_operator(gates, indices, Nb, No, operator, window_size, N=0):
    ex_factory = OperatorExampleIteratorFactory(indices, Nb, operator, N)
    packed_ex_factory = PackedExampleGenerator(ex_factory, No)
    return ChainedNetworkState(gates, packed_ex_factory, window_size)


cdef class NetworkState(BoolNetwork):
    cdef:
        readonly size_t Ne, cols
        packed_type_t[:, :] activation, inputs, outputs, target, error
        readonly packed_type_t zero_mask
        dict err_evaluators
        public size_t first_unevaluated_gate
        readonly bint evaluated

    def __init__(self, gates, size_t Ni, size_t No, size_t Ne, size_t cols):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        if 2**Ni < Ne:
            raise ValueError('More examples ({}) than #inputs ({}) '
                             'can represent.'.format(Ne, Ni))

        super().__init__(gates, Ni, No)

        self.Ne = Ne
        self.cols = cols

        self.zero_mask = generate_end_mask(Ne)

        # instantiate a matrix for activation
        self.activation = np.empty((self.Ng + Ni, cols), dtype=packed_type)

        # create input and output view into activation matrix
        self.inputs = self.activation[:Ni, :]
        self.outputs = self.activation[-<int>No:, :]

        self.target = np.empty((No, cols), dtype=packed_type)
        # instantiate matrices for error
        self.error = np.empty_like(self.target)

        self.err_evaluators = dict()

        self.evaluated = False
        self.first_unevaluated_gate = 0
        self._check_state_invariants()


    cpdef add_function(self, Function function):
        pass

    cpdef function_value(self, Function function):
        pass

    cpdef set_representation(self, np.uint32_t[:, :] network):
        # force reevaluation
        self.evaluated = False
        self.first_unevaluated_gate = 0
        BoolNetwork.set_representation(self, network)
        self._check_network_invariants()
        self._check_state_invariants()
        
    cpdef force_reevaluation(self):
        self.evaluated = False
        self.first_unevaluated_gate = 0
    
    cpdef randomise(self):
        BoolNetwork.randomise(self)
        # indicate the network must be reevaluated
        self.evaluated = False
        self.first_unevaluated_gate = 0

    cpdef apply_move(self, Move move):
        # indicate the network must be reevaluated
        self.evaluated = False
        if self.evaluated:
            self.first_unevaluated_gate = move.gate
        else:
            self.first_unevaluated_gate = min(self.first_unevaluated_gate, move.gate)
        BoolNetwork.apply_move(self, move)

    cpdef revert_move(self):
        cdef Move inverse_move
        inverse_move = self.inverse_moves.back()
        BoolNetwork.revert_move(self)
        if self.evaluated:
            self.first_unevaluated_gate = inverse_move.gate
        else:
            # if multiple moves are undone the network needs to
            # be recomputed from the earliest gate ever changed
            self.first_unevaluated_gate = min(
                self.first_unevaluated_gate, inverse_move.gate)
        self.evaluated = False

    cpdef revert_all_moves(self):
        BoolNetwork.revert_all_moves(self)
        self.first_unevaluated_gate = 0

    ############################### Evaluation methods ###############################
    cdef void _evaluate(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, src1, src2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint32_t[:, :] gates
            f_type func

        Ng = self.Ng
        Ni = self.Ni
        No = self.No
        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        outputs = self.outputs
        error = self.error
        target = self.target
        gates = self.gates
        
        cols = activation.shape[1]

        # evaluate the state matrix
        start = self.first_unevaluated_gate

        for g in range(start, Ng):
            src1 = gates[g, 0]
            src2 = gates[g, 1]
            func = function_list[gates[g, 2]]
            for c in range(cols):
                activation[Ni+g, c] = func(activation[src1, c], activation[src2, c])

        # evaluate the error matrix
        for o in range(No):
            for c in range(cols):
                error[o, c] = (target[o, c] ^ outputs[o, c])

    cdef _check_state_invariants(self):
        if self.cols == 0 or self.Ne == 0:
            raise ValueError('Zero value for cols ({}) or Ne ({}).'.
                             format(self.cols, self.Ne))
        if self.cols > self.Ne // PACKED_SIZE + 1:
            raise ValueError('Number of columns ({}) too large for Ne ({}) with PACKED_SIZE={}.'.
                             format(self.cols, self.Ne, PACKED_SIZE))
        if self.activation.shape[0] != self.Ni + self.Ng:
            raise ValueError('Activation length ({}) does not match Ni+Ng ({}).'.
                             format(self.activation.shape[0], self.Ni + self.Ng))
        if self.inputs.shape[0] != self.Ni:
            raise ValueError('Inputs length ({}) does not match Ni ({}).'.
                             format(self.inputs.shape[0], self.Ni))
        if self.outputs.shape[0] != self.No:
            raise ValueError('Outputs length ({}) does not match No ({}).'.
                             format(self.outputs.shape[0], self.No))
        if self.target.shape[0] != self.No:
            raise ValueError('Target length ({}) does not match No ({}).'.
                             format(self.target.shape[0], self.No))
        if self.error.shape[0] != self.No:
            raise ValueError('Error length ({}) does not match No ({}).'.
                             format(self.error.shape[0], self.No))
        shapes = [self.activation.shape[1], self.inputs.shape[1], self.target.shape[1],
                  self.outputs.shape[1], self.error.shape[1]]
        if not (min(shapes) == max(shapes) == self.cols):
            raise ValueError('Matrix column widths ({}) and cols not matching.'.
                             format(shapes, self.cols))


cdef class StandardNetworkState(NetworkState):

    def __init__(self, gates, packed_type_t[:, :] inputs,
                 packed_type_t[:, :] target, size_t Ne):
        if inputs.shape[1] != target.shape[1]:
            raise ValueError('Input ({}) and target ({}) widths do not match'.
                             format(inputs.shape[1], target.shape[1]))
        # if inputs.shape[0] != Ne:
        super().__init__(gates, inputs.shape[0], target.shape[0],
                         Ne, inputs.shape[1])
        self.inputs[...] = inputs
        self.target[...] = target

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
            return self.outputs

    property error_matrix:
        def __get__(self):
            self.evaluate()
            return self.error

    cpdef add_function(self, Function function):
        eval_class, msb = STANDARD_EVALUATORS[function]
        self.err_evaluators[function] = eval_class(self.Ne, self.No, msb)
        self.evaluated = False

    cpdef function_value(self, Function function):
        if function not in self.err_evaluators:
            self.add_function(function)
        if not self.evaluated:
            self.evaluate()
        return self.err_evaluators[function].evaluate(self.error)

    cpdef evaluate(self):
        ''' Evaluate the activation and error matrices if the
            network has been modified since the last evaluation. '''

        if self.evaluated:
            return

        self._evaluate()

        self._apply_zero_mask(self.activation)   # this does output_matrix as well (since it is a view)
        self._apply_zero_mask(self.error)

        self.evaluated = True

    cdef void _apply_zero_mask(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef size_t r, rows, cols

        rows, cols = matrix.shape[0], matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask


cdef class ChainedNetworkState(NetworkState):
    cdef:
        readonly PackedExampleGenerator example_generator
        size_t blocks, zero_mask_cols
        dict function_value_cache

    def __init__(self, gates, PackedExampleGenerator example_generator,
                 size_t window_size):
        super().__init__(gates, example_generator.Ni, example_generator.No,
                         example_generator.Ne, window_size)
        block_width = (self.cols * PACKED_SIZE)
        self.blocks = self.Ne // block_width
        if self.Ne % block_width:
            self.blocks += 1

        # work out the number of columns remaining after blocking
        # as this determines the zero_mask width
        total_cols = self.Ne // PACKED_SIZE
        if self.Ne % PACKED_SIZE:
            total_cols += 1 
        remaining_cols = total_cols % self.cols
        if remaining_cols > 0:
            self.zero_mask_cols = self.cols - remaining_cols + 1
        else:
            self.zero_mask_cols = 1
        self.example_generator = example_generator
        self.function_value_cache = dict()

    cpdef add_function(self, Function function):
        eval_class, msb = CHAINED_EVALUATORS[function]
        self.err_evaluators[function] = eval_class(self.Ne, self.No, self.cols, msb)
        self.evaluated = False

    cpdef function_value(self, Function function):
        if function not in self.err_evaluators:
            raise ValueError('Guiding function must be added to chained evaluator prior to evaluation.')

        if not self.evaluated:
            self.evaluate()
        return self.function_value_cache[function]

    cdef evaluate(self):
        cdef:
            size_t block
            dict evaluators = self.err_evaluators

        self.example_generator.reset()
        for m in evaluators:
            evaluators[m].reset()

        for block in range(self.blocks):
            self.example_generator.next_examples(self.inputs, self.target)
            self._evaluate()
            # on the last iteration we must not perform a partial evaluation
            if block < self.blocks - 1:
                for m in evaluators:
                    evaluators[m].partial_evaluation(self.error)

        self._apply_zero_mask(self.error)
        for m in evaluators:
            self.function_value_cache[m] = evaluators[m].final_evaluation(self.error)
        self.evaluated = True

    cdef void _apply_zero_mask(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef size_t r, c, rows, cols

        rows, cols = matrix.shape[0], matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-self.zero_mask_cols] &= self.zero_mask
            for c in range(1, self.zero_mask_cols):
                matrix[r, cols-self.zero_mask_cols+c] = 0

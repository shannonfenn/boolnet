# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
from copy import copy

from boolnet.network.boolnet cimport BoolNet, Move
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
#         return StandardBNState(network, mapping.inputs, mapping.target, mapping.Ne)
#     elif isinstance(mapping, OperatorBoolMapping):
#         return standard_from_operator(network, mapping.indices,
#                                       mapping.Nb, mapping.No,
#                                       mapping.operator, mapping.N)


#cpdef standard_from_operator(gates, indices, Nb, No, operator, N=0):
#    cdef packed_type_t[:, :] inp, tgt
#    inp, tgt = packed_from_operator(indices, Nb, No, operator, N)
#    return StandardBNState(gates, inp, tgt, Ne)


cpdef chained_from_operator(gates, indices, Nb, No, operator, window_size, N=0):
    ex_factory = OperatorExampleIteratorFactory(indices, Nb, operator, N)
    packed_ex_factory = PackedExampleGenerator(ex_factory, No)
    return ChainedBNState(gates, packed_ex_factory, window_size)


cdef class BNState:
    cdef:
        readonly BoolNet network
        readonly size_t Ne, cols
        packed_type_t[:, :] activation, inputs, outputs, target, error
        readonly packed_type_t zero_mask
        dict err_evaluators
        readonly size_t invalid_start
        readonly bint evaluated

    def __init__(self, gates, size_t Ni, size_t No, size_t Ne, size_t cols):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        if 2**Ni < Ne:
            print('WARNING: More examples ({}) than #inputs ({}) '
                  'can represent.'.format(Ne, Ni))

        self.network = BoolNet(gates, Ni, No)
        Ng = self.network.Ng

        self.Ne = Ne
        self.cols = cols

        self.zero_mask = generate_end_mask(Ne)

        # instantiate a matrix for activation
        self.activation = np.empty((Ng + Ni, cols), dtype=packed_type)

        # create input and output view into activation matrix
        self.inputs = self.activation[:Ni, :]
        self.outputs = self.activation[-<int>No:, :]

        self.target = np.empty((No, cols), dtype=packed_type)
        # instantiate matrices for error
        self.error = np.empty_like(self.target)

        self.err_evaluators = dict()

        self.evaluated = False
        self.invalid_start = 0
        self._check_state_invariants()

    property Ni:
        def __get__(self):
            return self.network.Ni

    property No:
        def __get__(self):
            return self.network.No

    property Ng:
        def __get__(self):
            return self.network.Ng

    property gates:
        def __get__(self):
            return self.network.gates

    cpdef add_function(self, Function function):
        pass

    cpdef function_value(self, Function function):
        pass

    cpdef set_gates(self, np.uint32_t[:, :] gates):
        # force reevaluation
        self.evaluated = False
        self.invalid_start = 0
        self.network.set_gates(gates)
        self._check_state_invariants()

    cpdef force_reevaluation(self):
        self.evaluated = False
        self.invalid_start = 0
    
    cpdef randomise(self):
        self.network.randomise()
        # indicate the network must be reevaluated
        self.evaluated = False
        self.invalid_start = 0

    cpdef move_to_random_neighbour(self):
        self.apply_move(self.network.random_move())

    cpdef apply_move(self, Move move):
        # indicate the network must be reevaluated
        self.evaluated = False
        if self.evaluated:
            self.invalid_start = move.gate
        else:
            self.invalid_start = min(self.invalid_start,
                                              move.gate)
        self.network.apply_move(move)

    cpdef revert_move(self):
        cdef Move inverse_move
        inverse_move = self.network.inverse_moves.back()
        self.network.revert_move()
        if self.evaluated:
            self.invalid_start = inverse_move.gate
        else:
            # if multiple moves are undone the network needs to
            # be recomputed from the earliest gate ever changed
            self.invalid_start = min(
                self.invalid_start, inverse_move.gate)
        self.evaluated = False

    cpdef revert_all_moves(self):
        self.network.revert_all_moves()
        self.evaluated = False
        self.invalid_start = 0

    ############################### Evaluation methods ###############################
    cdef void _evaluate(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, src1, src2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint32_t[:, :] gates
            f_type func

        Ng = self.network.Ng
        Ni = self.network.Ni
        No = self.network.No
        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        outputs = self.outputs
        error = self.error
        target = self.target
        gates = self.network.gates
        
        cols = activation.shape[1]

        # evaluate the state matrix
        start = self.invalid_start

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
        cdef size_t Ng, Ni, No
        Ng = self.network.Ng
        Ni = self.network.Ni
        No = self.network.No
        if self.cols == 0 or self.Ne == 0:
            raise ValueError('Zero value for cols ({}) or Ne ({}).'.
                             format(self.cols, self.Ne))
        if self.activation.shape[0] != Ni + Ng:
            raise ValueError('Activation length ({}) does not match Ni+Ng ({}).'.
                             format(self.activation.shape[0], Ni + Ng))
        if self.inputs.shape[0] != Ni:
            raise ValueError('Inputs length ({}) does not match Ni ({}).'.
                             format(self.inputs.shape[0], Ni))
        if self.outputs.shape[0] != No:
            raise ValueError('Outputs length ({}) does not match No ({}).'.
                             format(self.outputs.shape[0], No))
        if self.target.shape[0] != No:
            raise ValueError('Target length ({}) does not match No ({}).'.
                             format(self.target.shape[0], No))
        if self.error.shape[0] != No:
            raise ValueError('Error length ({}) does not match No ({}).'.
                             format(self.error.shape[0], self.No))
        if self.activation.shape[1] != self.cols:
            raise ValueError('Activation column width ({}) != cols ({}).'.
                             format(self.activation.shape[1], self.cols))
        if self.inputs.shape[1] != self.cols:
            raise ValueError('Inputs column width ({}) != cols ({}).'.
                             format(self.inputs.shape[1], self.cols))
        if self.outputs.shape[1] != self.cols:
            raise ValueError('Outputs column width ({}) != cols ({}).'.
                             format(self.outputs.shape[1], self.cols))
        if self.target.shape[1] != self.cols:
            raise ValueError('Target column width ({}) != cols ({}).'.
                             format(self.target.shape[1], self.cols))
        if self.error.shape[1] != self.cols:
            raise ValueError('Error column width ({}) != cols ({}).'.
                             format(self.error.shape[1], self.cols))


cdef class StandardBNState(BNState):

    def __init__(self, gates, packed_type_t[:, :] inputs,
                 packed_type_t[:, :] target, size_t Ne):
        if inputs.shape[1] != target.shape[1]:
            raise ValueError('Input ({}) and target ({}) widths do not match'.
                             format(inputs.shape[1], target.shape[1]))
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


cdef class ChainedBNState(BNState):
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

# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import cython
import numpy as np
cimport numpy as np
from copy import copy

from boolnet.network.boolnetwork import BoolNetwork
from boolnet.bintools.functions cimport Function
from boolnet.bintools.biterror import STANDARD_EVALUATORS
from boolnet.bintools.biterror cimport StandardEvaluator
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.biterror_chained cimport ChainedEvaluator
from boolnet.bintools.packing cimport packed_type_t, generate_end_mask, f_type, function_list, PACKED_SIZE
from boolnet.bintools.packing import packed_type
from boolnet.bintools.example_generator cimport PackedExampleGenerator, OperatorExampleIteratorFactory


cpdef standard_from_operator(network, indices, Nb, No, operator, N=0):
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
    return StandardNetworkState(network, inp, tgt, Ne)


cpdef chained_from_operator(network, indices, Nb, No, operator, window_size, N=0):
    ex_factory = OperatorExampleIteratorFactory(indices, Nb, operator, N)
    packed_ex_factory = PackedExampleGenerator(ex_factory, No)
    return ChainedNetworkState(network, packed_ex_factory, window_size)

        
cdef class NetworkState:
    cdef:
        readonly size_t Ne, Ni, No, Ng, cols
        packed_type_t[:, :] activation, inputs, outputs, target, error
        readonly packed_type_t zero_mask
        public object network
        dict err_evaluators

    def __init__(self, network, size_t Ne, size_t Ni, size_t No, size_t cols):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        if 2**Ni < Ne:
            raise ValueError('More examples ({}) than #inputs ({}) '
                             'can represent.'.format(Ne, Ni))
        self.Ne = Ne
        self.Ni = Ni
        self.No = No
        self.cols = cols

        self.zero_mask = generate_end_mask(Ne)

        # instantiate a matrix for activation
        self.activation = np.empty((network.Ng + self.Ni, self.cols), dtype=packed_type)

        # create input and output view into activation matrix
        self.inputs = self.activation[:self.Ni, :]
        self.outputs = self.activation[-<int>self.No:, :]

        self.target = np.empty((self.No, self.cols), dtype=packed_type)
        # instantiate matrices for error
        self.error = np.empty_like(self.target)

        self.set_network(network)
        self.err_evaluators = dict()

    cpdef add_function(self, Function function):
        pass

    cpdef function_value(self, Function function):
        pass

    ######################### Network pass-through methods #########################
    cpdef set_network(self, network):
        # check invariants hold
        self._check_network_invariants(network)
        self.Ng = network.Ng
        # prevent another evaluator causing problems with this network
        self.network = network.full_copy()
        # force reevaluation of the copied network
        self.network.changed = True
        self.network.first_unevaluated_gate = 0

    cpdef representation(self):
        return self.network

    cpdef move_to_random_neighbour(self):
        self.network.move_to_random_neighbour()

    cpdef revert_move(self):
        self.network.revert_move()

    cpdef clear_history(self):
        self.network.clear_history()

    ############################### Evaluation methods ###############################
    cdef void _evaluate_random(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint8_t[:] transfer_functions
            np.uint32_t[:, :] gates
            f_type transfer_func

        Ng = self.Ng
        Ni = self.Ni
        No = self.No
        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        outputs = self.outputs
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
            transfer_func = function_list[transfer_functions[g]]
            for c in range(cols):
                activation[Ni+g, c] = transfer_func(activation[in1, c], activation[in2, c])

        # evaluate the error matrix
        for o in range(No):
            for c in range(cols):
                error[o, c] = (target[o, c] ^ outputs[o, c])

    cdef void _evaluate_NAND(self):
        ''' Evaluate the activation and error matrices for the network
            assuming each node TF is NAND. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint32_t[:, :] gates
                
        Ng = self.Ng
        Ni = self.Ni
        No = self.No

        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        outputs = self.outputs
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
                error[o, c] = (target[o, c] ^ outputs[o, c])

    cdef _check_network_invariants(self, network):
        if self.Ng > 0 and network.Ng != self.Ng:
            raise ValueError(
                ('Network gate # ({}) does not match that of the network '
                'this evaluator was instantiatied with ({}).').format(network.Ng, self.Ng))
        if not isinstance(network, BoolNetwork):
            raise ValueError('\"network\" parameter not a subclass of \"BoolNetwork\"')
        if network.Ni != self.Ni:
            raise ValueError(('Network input # ({}) does not match input '
                             '({}).').format(network.No, self.Ni))
        if network.No != self.No:
            raise ValueError(('Network output # ({}) does not match target '
                             '({}).').format(network.No, self.No))


cdef class StandardNetworkState(NetworkState):

    def __init__(self, network, packed_type_t[:, :] inputs,
                 packed_type_t[:, :] target, size_t Ne):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        # check invariants hold
        self._check_instance_invariants(inputs, target, Ne)
        super().__init__(network, Ne, inputs.shape[0], target.shape[0], inputs.shape[1])
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
        self.network.changed = True

    cpdef function_value(self, Function function):
        if function not in self.err_evaluators:
            self.add_function(function)
        if self.network.changed:
            self.evaluate()
        return self.err_evaluators[function].evaluate(self.error)

    cpdef evaluate(self):
        ''' Evaluate the activation and error matrices if the
            network has been modified since the last evaluation. '''

        if not self.network.changed:
            return

        if hasattr(self.network, 'transfer_functions'):
            self._evaluate_random()
        else:
            self._evaluate_NAND()

        self._apply_zero_mask(self.activation)   # this does output_matrix as well (since it is a view)
        self._apply_zero_mask(self.error)

        self.network.changed = False

    cdef void _apply_zero_mask(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef:
            size_t r
            size_t rows = matrix.shape[0]
            size_t cols = matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask

    cdef _check_instance_invariants(self, packed_type_t[:, :] inputs,
                                    packed_type_t[:, :] target, size_t Ne):
        Ni = inputs.shape[0]
        No = target.shape[0]
        # Test if the matrices are valid
        if inputs.shape[1] == 0 or Ni == 0:
            raise ValueError('Empty input matrix.')
        if target.shape[1] == 0 or No == 0:
            raise ValueError('Empty target matrix.')
        if inputs.shape[1] != target.shape[1]:
            raise ValueError('Incompatible input/target shapes: {} {}.'.format(
                             inputs.shape[1], target.shape[1]))


cdef class ChainedNetworkState(NetworkState):
    cdef:
        readonly PackedExampleGenerator example_generator
        size_t blocks, zero_mask_cols
        dict function_value_cache
        bint evaluated

    def __init__(self, network, PackedExampleGenerator example_generator, size_t window_size):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        super().__init__(network, example_generator.Ne, example_generator.Ni,
                         example_generator.No, window_size)
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
        self.network.changed = True

    cpdef function_value(self, Function function):
        if function not in self.err_evaluators:
            raise ValueError('Guiding function must be added to evaluator prior to evaluation')

        if self.network.changed:
            self.evaluate()
        return self.function_value_cache[function]

    cdef evaluate(self):
        cdef:
            bint is_nand
            size_t block
            dict evaluators = self.err_evaluators

        is_nand = not hasattr(self.network, 'transfer_functions')
        self.example_generator.reset()
        for m in evaluators:
            evaluators[m].reset()

        for block in range(self.blocks):
            self.example_generator.next_examples(self.inputs, self.target)
            if is_nand:
                self._evaluate_NAND()
            else:
                self._evaluate_random()
            # on the last iteration we must not perform a partial evaluation
            if block < self.blocks - 1:
                for m in evaluators:
                    evaluators[m].partial_evaluation(self.error)

        self._apply_zero_mask(self.error)
        for m in evaluators:
            self.function_value_cache[m] = evaluators[m].final_evaluation(self.error)
        self.network.changed = False

    cdef void _apply_zero_mask(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef:
            size_t r, c
            size_t rows = matrix.shape[0]
            size_t cols = matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-self.zero_mask_cols] &= self.zero_mask
            for c in range(1, self.zero_mask_cols):
                matrix[r, cols-self.zero_mask_cols+c] = 0

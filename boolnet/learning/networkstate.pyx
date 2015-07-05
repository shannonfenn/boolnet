# cython: language_level=3
import cython
import numpy as np
cimport numpy as np
from copy import deepcopy

from boolnet.network.boolnetwork import BoolNetwork
from boolnet.bintools.biterror import STANDARD_EVALUATORS
from boolnet.bintools.biterror cimport StandardEvaluator
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.biterror_chained cimport ChainedEvaluator
from boolnet.bintools.packing cimport packed_type_t, generate_end_mask, f_type, function_list, PACKED_SIZE
from boolnet.bintools.packing import packed_type


cdef class StaticNetworkState:
    cdef:
        readonly size_t Ne, Ni, No, Ng, cols
        readonly metric
        packed_type_t[:, :] activation, inputs, outputs, target, error
        packed_type_t zero_mask
        StandardEvaluator err_evaluator
        public object network

    def __init__(self, network, packed_type_t[:, :] inputs,
                 packed_type_t[:, :] target, size_t Ne, metric=None):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        # check invariants hold
        self._check_instance_invariants(inputs, target, Ne)

        self.Ne = Ne
        self.Ni = inputs.shape[0]
        self.No = target.shape[0]
        self.cols = inputs.shape[1]

        # transpose and pack into integers
        self.target = np.array(target)

        self.zero_mask = generate_end_mask(Ne)
        
        # instantiate a matrix for activation
        self.activation = np.empty((network.Ng + self.Ni, inputs.shape[1]), dtype=packed_type)
        # copy inputs into activation matrix
        self.activation[:self.Ni, :] = inputs
        # create input and output view into activation matrix
        self.inputs = self.activation[:self.Ni, :]
        self.outputs = self.activation[-<int>self.No:, :]

        # instantiate matrices for error
        self.error = np.empty_like(self.target)

        self.set_network(network)
        self.set_metric(metric)

    def set_network(self, network):
        # check invariants hold
        self._check_network_invariants(network)
        self.Ng = network.Ng
        # prevent another evaluator causing problems with this network
        self.network = deepcopy(network)
        # force reevaluation of the copied network
        self.network._evaluated = False
        self.network.first_unevaluated_gate = 0

    def set_metric(self, metric):
        self.metric = metric
        if metric is not None:
            eval_class, msb = STANDARD_EVALUATORS[metric]
            self.err_evaluator = eval_class(self.Ne, self.No, msb)
        else:
            self.err_evaluator = None


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

    def metric_value(self):
        self.evaluate()
        return self.err_evaluator.evaluate(self.error)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _zero_mask_last_column(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef:
            size_t r
            size_t rows = matrix.shape[0]
            size_t cols = matrix.shape[1]

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
            packed_type_t[:, :] activation, outputs, error, target
            np.uint8_t[:] transfer_functions
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
                error[o, c] = (target[o, c] ^ outputs[o, c])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
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

    cdef _check_instance_invariants(self, packed_type_t[:, :] inputs,
                                    packed_type_t[:, :] target, size_t Ne):
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

cdef class ChainedNetworkState:
    cdef:
        readonly size_t Ne, Ni, No, Ng, cols
        readonly metric
        packed_type_t[:, :] activation, inputs, outputs,
        packed_type_t[:, :] target, error
        packed_type_t zero_mask
        ChainedEvaluator err_evaluator
        public object network

    def __init__(self, network, example_generator, window_size, metric=None):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('Invalid window_size.')

        self.Ne = example_generator.Ne
        self.Ni = example_generator.Ni
        self.No = example_generator.No
        self.cols = window_size
        self.block_width = (self.cols * PACKED_SIZE)
        if self.Ne % self.block_width == 0:
            self.blocks = self.Ne // self.block_width
        else:
            self.blocks = self.Ne // self.block_width + 1

        self.zero_mask = generate_end_mask(self.Ne)
        
        self.target = np.empty((self.No, self.cols), dtype=packed_type)
        # instantiate a matrix for activation
        self.activation = np.empty((network.Ng + self.Ni, self.cols), dtype=packed_type)
        # create input and output view into activation matrix
        self.inputs = self.activation[:self.Ni, :]
        self.outputs = self.activation[-<int>self.No:, :]

        # instantiate matrices for error
        self.error = np.empty_like(self.target)
        self.error_scratch = np.empty_like(self.target)

        self.set_network(network)
        self.set_metric(metric)

    def set_network(self, network):
        # check invariants hold
        self._check_network_invariants(network)
        self.Ng = network.Ng
        # prevent another evaluator causing problems with this network
        self.network = deepcopy(network)
        # force reevaluation of the copied network
        self.network._evaluated = False
        self.network.first_unevaluated_gate = 0

    def set_metric(self, metric):
        self.metric = metric
        if metric is not None:
            eval_class, msb = CHAINED_EVALUATORS[metric]
            self.err_evaluator = eval_class(self.Ne, self.No, msb)
        else:
            self.err_evaluator = None

    def metric_value(self):
        # THIS IS WHERE THE MAGIC NEEDS TO HAPPEN

        # LOOP
        #   load next inp / tgt pair
        #   evaluate network
        #   partialevaluate metric value
        # END LOOP
        # finalise metric value
        self.evaluate()
        return self.err_evaluator.evaluate(self.error)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _zero_mask_last_column(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef:
            size_t r
            size_t rows = matrix.shape[0]
            size_t cols = matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask

    cpdef evaluate(self):
        ''' Evaluate the activation and error matrices if the
            network has been modified since the last evaluation. '''
        if hasattr(self.network, 'transfer_functions'):
            self._evaluate_random()
        else:
            self._evaluate_NAND()

        self._zero_mask_last_column(self.activation)   # this does output_matrix as well (since it is a view)
        self._zero_mask_last_column(self.error)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _evaluate_random(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, in1, in2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint8_t[:] transfer_functions
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
                error[o, c] = (target[o, c] ^ outputs[o, c])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
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

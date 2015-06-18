import numpy as np
import BoolNet.BitError as BitError
import collections
from copy import deepcopy


State = collections.namedtuple('State', [
    'network', 'activation', 'output', 'error'])


def _check_combined_invariants(network, inputs, target):
    ''' [TODO] Add more here. '''
    if network.Ng < target.shape[1]:
        raise ValueError(('Not enough gates in network ({}) to express target '
                         '({}).').format(network.Ng, target.shape[1]))


def _check_instance_invariants(inputs, target):
    # Test if we have 2D matrices
    if inputs.ndim != 2 or target.ndim != 2:
        raise ValueError('Invalid input ({0}) or target ({1}) '
                         'dimension number.'.format(inputs.ndim,
                                                    target.ndim))
    Ne, Ni = inputs.shape
    No = target.shape[1]
    # Test if the matrices are valid
    if Ne == 0 or Ni == 0:
        raise ValueError('Empty input matrix.')
    if target.shape[0] == 0 or No == 0:
        raise ValueError('Empty target matrix.')
    if 2**Ni < Ne:
        raise ValueError('More examples ({}) than #inputs ({}) '
                         'can represent.'.format(*inputs.shape))
    if Ne != target.shape[0]:
        raise ValueError('Incompatible input/target shapes: {} {}.'.format(
                         inputs.shape, target.shape))


class NetworkEvaluator:

    def __init__(self, inputs, target):
        self._inputs = np.array(inputs, dtype=np.uint8)
        self._target = np.array(target, dtype=np.uint8)
        # add any given networks into states array
        self._states = []
        # check invariants hold
        _check_instance_invariants(self._inputs, self._target)

    def __str__(self):
        return ('Ne: {} Ni: {} No: {} inputs: {} target: {} '
                'networks: {} activation matrices: {} '
                'error matrices: {}').format(
            self._inputs.shape[0], self._inputs.shape[1],
            self._target.shape[1], self._inputs, self._target,
            list(s.network for s in self._states),
            list(s.activation for s in self._states),
            list(s.error for s in self._states))

    @property
    def input_matrix(self):
        return self._inputs

    @property
    def target_matrix(self):
        return self._target

    @property
    def num_networks(self):
        return len(self._states)

    def network(self, index):
        return self._states[index].network

    def output_matrix(self, index):
        self.evaluate(index)
        return self._states[index].output

    def error_matrix(self, index):
        self.evaluate(index)
        return self._states[index].error

    def activation_matrix(self, index):
        self.evaluate(index)
        return self._states[index].activation

    def add_network(self, network):
        ''' Set up the activation and error matrices for a new network and add the
            resulting State struct to the states list.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        _check_combined_invariants(network, self._inputs, self._target)

        Ne, Ni = self._inputs.shape
        No = self._target.shape[1]
        Ng = network.Ng

        # instantiate a matrix for activation and for error
        activation_matrix = np.empty((Ne, Ni+Ng), dtype=np.uint8)
        error_matrix = np.empty(self._target.shape, dtype=np.uint8)

        # copy inputs into activation matrix
        activation_matrix[:, :Ni] = self._inputs

        # construct view into activation for output matrix
        output_matrix = activation_matrix[:, -No:]

        # this prevents another evaluator causing problems with this network
        network = deepcopy(network)
        # indicate that the network needs to be evaluated later
        network._evaluated = False
        network.first_unevaluated_gate = 0

        # append evaluation struct to list of states
        self._states.append(State(network, activation_matrix,
                                  output_matrix, error_matrix))

    def remove_network(self, index):
        self._states.pop(index)

    def remove_all_networks(self):
        self._states = []

    def metric_value(self, index, metric):
        self.evaluate(index)
        return BitError.metric_value(self._states[index].error, metric)

    def evaluate(self, index):
        ''' Evaluate the activation and error matrices for the selected network if it
            has been modified since the last evaluation. '''
        state = self._states[index]

        if state.network._evaluated:
            return

        if hasattr(state.network, 'transfer_functions'):
            self._evaluate_random(state)
        else:
            self._evaluate_NAND(state)

        state.network._evaluated = True

    def _evaluate_random(self, state):
        ''' Evaluate the activation and error matrices for the selected network
            getting node TFs from network. '''
        Ni = self._inputs.shape[1]

        activation = state.activation
        TF = state.network.transfer_functions

        # evaluate the state matrix
        gates = state.network.gates
        for g in range(state.network.Ng):
            gate = gates[g]
            activation[:, Ni + g] = TF[
                g, activation[:, gate[0]], activation[:, gate[1]]]

        # evaluate the error matrix
        np.not_equal(state.output, self._target, state.error)

    def _evaluate_NAND(self, state):
        ''' Evaluate the activation and error matrices for the selected network
            assuming each node TF is NAND. '''
        Ni = self._inputs.shape[1]

        activation = state.activation

        # evaluate the state matrix
        gates = state.network.gates
        for g in range(state.network.Ng):
            gate = gates[g]
            np.logical_not(np.logical_and(activation[:, gate[0]],
                                          activation[:, gate[1]]),
                           activation[:, Ni + g])
            # np.logical_not(activation[:, gate].all(),
            #                activation[:, Ni + g])

        # evaluate the error matrix
        np.not_equal(state.output, self._target, state.error)

    def truth_table(self, index):
        ''' Generate and return the full truth table for the chosen network.
            WARNING: This is exponentially large with Ni. '''
        state = self._states[index]

        if hasattr(state.network, 'transfer_functions'):
            return self._tt_random(state)
        else:
            return self._tt_NAND(state)

    def _tt_random(self, state):
        ''' Generate and return the full truth table for the chosen network
            getting node TFs from network.
            WARNING: This is exponentially large with Ni. '''
        Ni = self._inputs.shape[1]
        No = self._target.shape[1]
        gates = state.network.gates
        TF = state.network.transfer_functions

        activation = np.zeros(Ni + state.network.Ng, dtype=np.uint8)
        output_table = np.empty((2**Ni, No), dtype=np.uint8)

        for i in range(2**Ni):
            # Generate next input
            activation[:Ni] = [(1 << b & i) != 0 for b in range(Ni)]
            # evaluate state vector
            for g, gate in enumerate(gates):
                activation[Ni + g] = TF[g, activation[gate[0]], activation[gate[1]]]
            # copy output states to table
            output_table[i, :] = activation[-No:]

        return output_table

    def _tt_NAND(self, state):
        ''' Generate and return the full truth table for the chosen network
            assuming each node TF is NAND.
            WARNING: This is exponentially large with Ni. '''
        Ni = self._inputs.shape[1]
        No = self._target.shape[1]
        gates = state.network.gates

        activation = np.zeros(Ni + state.network.Ng, dtype=np.uint8)
        output_table = np.empty((2**Ni, No), dtype=np.uint8)

        for i in range(2**Ni):
            # Generate next input
            activation[:Ni] = [(1 << b & i) != 0 for b in range(Ni)]
            # evaluate state vector
            for g, gate in enumerate(gates):
                activation[Ni + g] = not(activation[gate[0]] and
                                         activation[gate[1]])
            # copy output states to table
            output_table[i, :] = activation[-No:]

        return output_table

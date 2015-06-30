from collections import namedtuple
from BoolNet.Packing import pack_bool_matrix, unpack_bool_matrix
from BoolNet.boolnetwork import BoolNetwork
import numpy as np
import pytest

import pyximport
pyximport.install()


Move = namedtuple('Move', ['gate', 'terminal', 'source'])


# ############# General helpers ################# #
def to_binary(value, num_bits):
    return [int(i) for i in '{:0{w}b}'.format(value, w=num_bits)][::-1]


def all_possible_inputs(num_bits):
    return pack_bool_matrix(np.array(
        [to_binary(i, num_bits) for i in range(2**num_bits)],
        dtype=np.byte))


def packed_zeros(shape):
    return pack_bool_matrix(np.zeros(shape, dtype=np.uint8))


# ################### Exception Testing ################### #
@pytest.mark.parametrize("net, Ni, Tshape, Ne", [
    (BoolNetwork([(0, 1)], 1, 1), 1, (2**4, 1), 2**4),  # Ne != inputs.Ne
    (BoolNetwork([(0, 1)], 1, 1), 4, (2, 1), 2**4),     # Ne != target.Ne
    (BoolNetwork([(0, 1)], 1, 1), 1, (2, 4), 4),        # net.No != #targets
    (BoolNetwork([(0, 1)], 1, 1), 2, (1, 4), 4),        # net.Ni != #inputs
    (BoolNetwork([(0, 1)], 1, 1), 2, (2, 4), 4)         # both
])
def test_construction_exceptions(net, Ni, Tshape, Ne, evaluator_class):
    inp = all_possible_inputs(Ni)
    tgt = packed_zeros(Tshape)
    with pytest.raises(ValueError):
        evaluator_class(net, inp, tgt, Ne)


# ################### Functionality Testing ################### #
class TestFunctionality:

    def test_input_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.input_matrix, Ne)
        Ni = any_test_network['Ni']
        activation = np.array(any_test_network['activation matrix'][network_type], dtype=np.uint8)
        expected = activation[:, :Ni]
        assert np.array_equal(actual, expected)

    def test_target_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.target_matrix, Ne)
        expected = np.array(any_test_network['target matrix'][network_type], dtype=np.uint8)
        assert np.array_equal(actual, expected)

    def test_output_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        No = any_test_network['No']

        activation = np.array(any_test_network['activation matrix'][network_type], dtype=np.uint8)
        expected = activation[:, -No:]
        actual = unpack_bool_matrix(evaluator.output_matrix, Ne)

        assert np.array_equal(actual, expected)

    # def test_truth_table(self, any_test_network, network_type):
    #     evaluator = any_test_network['evaluator'][network_type]
    #     Ne = any_test_network['target matrix'][network_type].shape[0]
    #     actual = unpack_bool_matrix(evaluator.truth_table(0), Ne)
    #     activation = np.array(any_test_network['activation matrix']['full'], dtype=np.uint8)
    #     expected = activation[:, -any_test_network['No']:]
    #     assert np.array_equal(actual, expected)

    def test_activation_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.activation_matrix, Ne)
        expected = np.array(
            any_test_network['activation matrix'][network_type],
            dtype=np.byte)
        assert np.array_equal(actual, expected)

    def test_single_move_output_different(
            self, single_move_invariant, network_type):
        evaluator = single_move_invariant['evaluator'][network_type]
        net = evaluator.network
        Ne = single_move_invariant['target matrix'][network_type].shape[0]
        for k in range(100):
            old_error = np.array(evaluator.error_matrix, copy=True)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(
                unpack_bool_matrix(evaluator.error_matrix, Ne),
                old_error)
            net.revert_move()

    def test_multiple_move_output_different(
            self, multiple_move_invariant, network_type):
        evaluator = multiple_move_invariant['evaluator'][network_type]
        net = evaluator.network
        Ne = multiple_move_invariant['target matrix'][network_type].shape[0]
        for k in range(100):
            old_error = np.array(evaluator.error_matrix, copy=True)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(
                unpack_bool_matrix(evaluator.error_matrix, Ne),
                old_error)

    def test_move_with_initial_evaluation(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network
        Ne = single_layer_zero['target matrix'][network_type].shape[0]

        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = single_layer_zero['error matrix'][network_type]

        assert np.array_equal(actual, expected)

        move = Move(gate=1, source=4, terminal=True)

        net.move_to_neighbour(move)
        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = np.array([
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1],
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.byte)
        assert np.array_equal(actual, expected)

    def test_multiple_moves_error_matrix(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network

        moves = [Move(gate=0, source=1, terminal=False),
                 Move(gate=1, source=0, terminal=True),
                 Move(gate=0, source=0, terminal=True),
                 Move(gate=1, source=3, terminal=True),
                 Move(gate=1, source=4, terminal=True),
                 Move(gate=1, source=4, terminal=False)]

        expected = np.array([
            # expected after move 1
            [[1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [0, 1], [0, 1],
             [1, 1], [1, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0]],
            # expected after move 2
            [[1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0],
             [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0]],
            # expected after move 3
            [[1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]],
            # expected after move 4
            [[1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1], [0, 1],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 0]],
            # expected after move 5
            [[1, 1],  [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]],
            # expected after move 6
            [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
             [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]]],
            dtype=np.byte)

        Ne = single_layer_zero['target matrix'][network_type].shape[0]
        for move, expectation in zip(moves, expected):
            net.move_to_neighbour(move)
            actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
            assert np.array_equal(actual, expectation)

    def test_multiple_reverts_error_matrix(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network

        moves = [Move(gate=0, source=1, terminal=False),
                 Move(gate=1, source=0, terminal=True),
                 Move(gate=0, source=0, terminal=True),
                 Move(gate=1, source=3, terminal=True),
                 Move(gate=1, source=4, terminal=True),
                 Move(gate=1, source=4, terminal=False)]

        expected = np.array([
            # expected after move 1
            [[1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [0, 1], [0, 1],
             [1, 1], [1, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0]],
            # expected after move 2
            [[1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0],
             [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0]],
            # expected after move 3
            [[1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]],
            # expected after move 4
            [[1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1], [0, 1],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 0]],
            # expected after move 5
            [[1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
             [1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]],
            # expected after move 6
            [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
             [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]]],
            dtype=np.byte)

        for move in moves:
            net.move_to_neighbour(move)

        Ne = single_layer_zero['target matrix'][network_type].shape[0]
        for expectation in reversed(expected):
            actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
            assert np.array_equal(actual, expectation)
            net.revert_move()

    def test_error_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = any_test_network['error matrix'][network_type]
        assert np.array_equal(actual, expected)

    def test_metric_value(self, any_test_network, metric, network_type):
        # a value may not have been specified for all metrics
        # on this test network
        evaluator = any_test_network['evaluator'][network_type]
        actual = evaluator.metric_value(metric)
        metric_values = any_test_network['metric value'][network_type]
        expected = metric_values[str(metric)]
        assert np.array_equal(actual, expected)

    def test_pre_evaluated_network(self, any_test_network):
        evaluator1 = any_test_network['evaluator']['sample']
        evaluator2 = any_test_network['evaluator']['full']
        Ne1 = any_test_network['target matrix']['sample'].shape[0]
        Ne2 = any_test_network['target matrix']['full'].shape[0]

        net = evaluator1.network
        for i in range(10):
            net.move_to_neighbour(net.random_move())
            evaluator1.evaluate()
        net.revert_all_moves()

        # check sample evaluator is still giving original results
        expected = unpack_bool_matrix(evaluator1.activation_matrix, Ne1)
        actual = any_test_network['activation matrix']['sample']
        assert np.array_equal(expected, actual)

        # check full evaluator is still giving original results
        evaluator2.set_network(net)
        expected = unpack_bool_matrix(evaluator2.activation_matrix, Ne2)
        actual = any_test_network['activation matrix']['full']
        assert np.array_equal(expected, actual)

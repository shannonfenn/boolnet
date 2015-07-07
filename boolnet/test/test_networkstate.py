from collections import namedtuple
from boolnet.bintools.packing import pack_bool_matrix, unpack_bool_matrix
from boolnet.network.boolnetwork import BoolNetwork
import numpy as np
from numpy.testing import assert_array_equal as assert_array_equal
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


# ################### Functionality Testing ################### #
class TestStandard:

    # ################### Exception Testing ################### #
    @pytest.mark.parametrize("net, Ni, Tshape, Ne", [
        (BoolNetwork([(0, 1)], 1, 1), 1, (2**4, 1), 2**4),  # Ne != inputs.Ne
        (BoolNetwork([(0, 1)], 1, 1), 4, (2, 1), 2**4),     # Ne != target.Ne
        (BoolNetwork([(0, 1)], 1, 1), 1, (2, 4), 4),        # net.No != #targets
        (BoolNetwork([(0, 1)], 1, 1), 2, (1, 4), 4),        # net.Ni != #inputs
        (BoolNetwork([(0, 1)], 1, 1), 2, (2, 4), 4)         # both
    ])
    def test_construction_exceptions(self, net, Ni, Tshape, Ne, evaluator_class):
        inp = all_possible_inputs(Ni)
        tgt = packed_zeros(Tshape)
        with pytest.raises(ValueError):
            evaluator_class(net, inp, tgt, Ne)

    def build_instance(self, instance_dict, network_type, field):
        evaluator = instance_dict['evaluator'][network_type]
        Ne = instance_dict['Ne'][network_type]
        expected = np.array(instance_dict[field][network_type], dtype=np.uint8)
        eval_func = lambda mat: unpack_bool_matrix(mat, Ne)
        return evaluator, expected, eval_func

    def test_input_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'input matrix')
        actual = eval_func(evaluator.input_matrix)
        assert_array_equal(actual, expected)

    def test_target_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'target matrix')
        actual = eval_func(evaluator.target_matrix)
        assert_array_equal(actual, expected)

    def test_output_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'output matrix')
        actual = eval_func(evaluator.output_matrix)
        assert_array_equal(actual, expected)

    def test_activation_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'activation matrix')
        actual = eval_func(evaluator.activation_matrix)
        assert_array_equal(actual, expected)

    def output_different_helper(self, instance, network_type):
        evaluator, _, eval_func = self.build_instance(instance, network_type,
                                                      'error matrix')
        net = evaluator.network
        for k in range(10):
            old_error = eval_func(evaluator.error_matrix)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(eval_func(evaluator.error_matrix), old_error)
            net.revert_move()

    def test_single_move_output_different(
            self, single_move_invariant, network_type):
        self.output_different_helper(single_move_invariant, network_type)

    def test_multiple_move_output_different(
            self, multiple_move_invariant, network_type):
        self.output_different_helper(multiple_move_invariant, network_type)

    def test_move_with_initial_evaluation(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network
        Ne = single_layer_zero['target matrix'][network_type].shape[0]

        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = single_layer_zero['error matrix'][network_type]

        assert_array_equal(actual, expected)

        move = Move(gate=1, source=4, terminal=True)

        net.move_to_neighbour(move)
        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = np.array([
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1],
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.byte)
        assert_array_equal(actual, expected)

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
            assert_array_equal(actual, expectation)

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
            assert_array_equal(actual, expectation)
            net.revert_move()

    def test_error_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.error_matrix, Ne)
        expected = any_test_network['error matrix'][network_type]
        assert_array_equal(actual, expected)

    def test_metric_value(self, any_test_network, metric, network_type):
        # a value may not have been specified for all metrics
        # on this test network
        evaluator = any_test_network['evaluator'][network_type]
        evaluator.set_metric(metric)
        actual = evaluator.metric_value()
        metric_values = any_test_network['metric value'][network_type]
        expected = metric_values[str(metric)]
        assert_array_equal(actual, expected)

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
        assert_array_equal(expected, actual)

        # check full evaluator is still giving original results
        evaluator2.set_network(net)
        expected = unpack_bool_matrix(evaluator2.activation_matrix, Ne2)
        actual = any_test_network['activation matrix']['full']
        assert_array_equal(expected, actual)

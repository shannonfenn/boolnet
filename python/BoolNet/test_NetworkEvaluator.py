from copy import copy
from collections import namedtuple
from BoolNet.Packing import pack_bool_matrix, unpack_bool_matrix
from BoolNet.BooleanNetwork import BooleanNetwork
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


Instance = namedtuple('Instance', ['inputs', 'target', 'Ne'])


class TestExceptions:
    @pytest.fixture
    def gates(self):
        return np.array([(0, 1)], dtype=np.uint32),

    @pytest.fixture
    def instance(self):
        # gates
        return Instance(
            inputs=all_possible_inputs(4),
            target=pack_bool_matrix(np.zeros((2**4, 1), dtype=np.uint8)),
            Ne=2**4)

    ''' Exception Testing '''
    def test_construction_exceptions(self, instance, evaluator_class):
        ''' This test may not check all cases but is not
            particularly important as it is just error checks.'''
        # num input examples != num target examples
        with pytest.raises(ValueError):
            evaluator_class(all_possible_inputs(1),
                            instance.target, instance.Ne)
        with pytest.raises(ValueError):
            evaluator_class(instance.inputs, np.zeros((2, 1),
                            dtype=np.byte), instance.Ne)

    def test_addition_exceptions(self, gates, evaluator_class):
        ''' This test may not check all cases but is not
            particularly important as it is just error checks.'''
        # num input examples != num target examples
        e = evaluator_class(all_possible_inputs(2),
                            pack_bool_matrix(np.zeros((4, 2), dtype=np.uint8)),
                            4)
        with pytest.raises(ValueError):
            e.add_network(BooleanNetwork(gates, 1, 1))


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

        print(evaluator.activation_matrix(0))
        print(evaluator.output_matrix(0))
        print(unpack_bool_matrix(evaluator.output_matrix(0), Ne))

        actual = unpack_bool_matrix(evaluator.output_matrix(0), Ne)

        print(actual)
        print(expected)

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
        actual = unpack_bool_matrix(evaluator.activation_matrix(0), Ne)
        expected = np.array(
            any_test_network['activation matrix'][network_type],
            dtype=np.byte)
        assert np.array_equal(actual, expected)

    def test_single_move_output_different(
            self, single_move_invariant, network_type):
        evaluator = single_move_invariant['evaluator'][network_type]
        net = evaluator.network(0)
        Ne = single_move_invariant['target matrix'][network_type].shape[0]
        for k in range(100):
            old_error = np.array(evaluator.error_matrix(0), copy=True)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(
                unpack_bool_matrix(evaluator.error_matrix(0), Ne),
                old_error)
            net.revert_move()

    def test_multiple_move_output_different(
            self, multiple_move_invariant, network_type):
        evaluator = multiple_move_invariant['evaluator'][network_type]
        net = evaluator.network(0)
        Ne = multiple_move_invariant['target matrix'][network_type].shape[0]
        for k in range(100):
            old_error = np.array(evaluator.error_matrix(0), copy=True)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(
                unpack_bool_matrix(evaluator.error_matrix(0), Ne),
                old_error)

    def test_move_with_initial_evaluation(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network(0)
        Ne = single_layer_zero['target matrix'][network_type].shape[0]

        actual = unpack_bool_matrix(evaluator.error_matrix(0), Ne)
        expected = single_layer_zero['error matrix'][network_type]

        assert np.array_equal(actual, expected)

        move = Move(gate=1, source=4, terminal=True)

        net.move_to_neighbour(move)
        actual = unpack_bool_matrix(evaluator.error_matrix(0), Ne)
        expected = np.array([
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1],
            [1, 1], [1, 1], [1, 1], [0, 1],
            [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.byte)
        assert np.array_equal(actual, expected)

    def test_multiple_moves_error_matrix(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network(0)

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
            actual = unpack_bool_matrix(evaluator.error_matrix(0), Ne)
            assert np.array_equal(actual, expectation)

    def test_multiple_reverts_error_matrix(
            self, single_layer_zero, network_type):
        evaluator = single_layer_zero['evaluator'][network_type]
        net = evaluator.network(0)

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
            actual = unpack_bool_matrix(evaluator.error_matrix(0), Ne)
            assert np.array_equal(actual, expectation)
            net.revert_move()

    def test_error_matrix(self, any_test_network, network_type):
        evaluator = any_test_network['evaluator'][network_type]
        Ne = any_test_network['target matrix'][network_type].shape[0]
        actual = unpack_bool_matrix(evaluator.error_matrix(0), Ne)
        expected = any_test_network['error matrix'][network_type]
        assert np.array_equal(actual, expected)

    def test_metric_value(self, any_test_network, metric, network_type):
        # a value may not have been specified for all metrics
        # on this test network
        evaluator = any_test_network['evaluator'][network_type]
        actual = evaluator.metric_value(0, metric)
        metric_values = any_test_network['metric value'][network_type]
        expected = metric_values[str(metric)]
        assert np.array_equal(actual, expected)

    def test_pre_evaluated_network(self, any_test_network, network_type):
        if network_type == 'sample':
            evaluator1 = any_test_network['evaluator']['sample']
            evaluator2 = any_test_network['evaluator']['full']
            evaluator1.remove_all_networks()
            evaluator2.remove_all_networks()
            net = any_test_network['network']
            evaluator1.add_network(net)
            for i in range(10):
                net.move_to_neighbour(net.random_move())
                evaluator1.evaluate(0)
            net.revert_all_moves()
            Ne = any_test_network['target matrix']['sample'].shape[0]
            assert np.array_equal(
                unpack_bool_matrix(evaluator1.activation_matrix(0), Ne),
                any_test_network['activation matrix']['sample'])
            evaluator2.add_network(net)
            Ne = any_test_network['target matrix']['full'].shape[0]
            assert np.array_equal(
                unpack_bool_matrix(evaluator2.activation_matrix(0), Ne),
                any_test_network['activation matrix']['full'])

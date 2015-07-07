from boolnet.bintools.packing import pack_bool_matrix, unpack_bool_matrix
from boolnet.network.boolnetwork import BoolNetwork
import numpy as np
from numpy.testing import assert_array_equal as assert_array_equal
import pytest

import pyximport
pyximport.install()


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
        assert_array_equal(expected, actual)

    def test_target_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'target matrix')
        actual = eval_func(evaluator.target_matrix)
        assert_array_equal(expected, actual)

    def test_output_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'output matrix')
        actual = eval_func(evaluator.output_matrix)
        assert_array_equal(expected, actual)

    def test_activation_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'activation matrix')
        actual = eval_func(evaluator.activation_matrix)
        assert_array_equal(expected, actual)

    def test_error_matrix(self, any_test_network, network_type):
        evaluator, expected, eval_func = self.build_instance(
            any_test_network, network_type, 'error matrix')
        actual = eval_func(evaluator.error_matrix)
        assert_array_equal(expected, actual)

    def test_metric_value(self, any_test_network, metric, network_type):
        # a value may not have been specified for all metrics
        # on this test network
        evaluator = any_test_network['evaluator'][network_type]
        evaluator.set_metric(metric)
        expected = any_test_network['metric value'][network_type][str(metric)]
        actual = evaluator.metric_value()
        assert_array_equal(expected, actual)

    def output_different_helper(self, instance, network_type):
        evaluator, _, eval_func = self.build_instance(instance, network_type,
                                                      'error matrix')
        net = evaluator.network
        for k in range(10):
            old_error = eval_func(evaluator.error_matrix)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(eval_func(evaluator.error_matrix), old_error)
            net.revert_move()

    def test_single_move_output_different(self, single_move_invariant, network_type):
        self.output_different_helper(single_move_invariant, network_type)

    def test_multiple_move_output_different(self, multiple_move_invariant, network_type):
        self.output_different_helper(multiple_move_invariant, network_type)

    def test_move_with_initial_evaluation(self, single_layer_zero, network_type):
        evaluator, expected, eval_func = self.build_instance(
            single_layer_zero, network_type, 'error matrix')

        net = evaluator.network

        actual = eval_func(evaluator.error_matrix)
        assert_array_equal(expected, actual)

        test_case = single_layer_zero['multiple_moves_test_case'][4]

        net.move_to_neighbour(test_case.move)
        actual = eval_func(evaluator.error_matrix)
        expected = test_case.expected
        assert_array_equal(expected, actual)

    def test_multiple_moves_error_matrix(self, single_layer_zero, network_type):
        evaluator, _, eval_func = self.build_instance(
            single_layer_zero, network_type, 'error matrix')
        net = evaluator.network

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, expected in test_case:
            net.move_to_neighbour(move)
            actual = eval_func(evaluator.error_matrix)
            assert_array_equal(expected, actual)

    def test_multiple_reverts_error_matrix(self, single_layer_zero, network_type):
        evaluator, _, eval_func = self.build_instance(
            single_layer_zero, network_type, 'error matrix')

        net = evaluator.network

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, _ in test_case:
            net.move_to_neighbour(move)

        for _, expected in reversed(test_case):
            actual = eval_func(evaluator.error_matrix)
            assert_array_equal(expected, actual)
            net.revert_move()

    def test_pre_evaluated_network(self, any_test_network):
        evaluator_s, expected_s, eval_func_s = self.build_instance(
            any_test_network, 'sample', 'activation matrix')
        evaluator_f, expected_f, eval_func_f = self.build_instance(
            any_test_network, 'full', 'activation matrix')

        net = evaluator_s.network
        for i in range(10):
            net.move_to_neighbour(net.random_move())
            evaluator_s.evaluate()
        net.revert_all_moves()

        # check sample evaluator is still giving original results
        actual = eval_func_s(evaluator_s.activation_matrix)
        assert_array_equal(expected_s, actual)

        # check full evaluator is still giving original results
        evaluator_f.set_network(net)
        actual = eval_func_f(evaluator_f.activation_matrix)
        assert_array_equal(expected_f, actual)

import yaml
import glob
import numpy as np
from copy import copy, deepcopy
from collections import namedtuple
from pytest import mark, raises, fixture
from numpy.testing import assert_array_equal as assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_array_almost_equal

import pyximport
pyximport.install()
from boolnet.bintools.packing import pack_bool_matrix, unpack_bool_matrix, generate_end_mask
from boolnet.bintools.packing import PACKED_SIZE_PY as PACKED_SIZE
from boolnet.bintools.metrics import metric_name, all_metrics
from boolnet.bintools.example_generator import PackedExampleGenerator, OperatorExampleFactory
from boolnet.bintools.operator_iterator import ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from boolnet.learning.networkstate import (StandardNetworkState, ChainedNetworkState,
                                           standard_from_operator, chained_from_operator)


TEST_NETWORKS = glob.glob('boolnet/test/networks/*.yaml')


operator_map = {
    'zero': ZERO, 'and': AND, 'or': OR,
    'unary_and': UNARY_AND, 'unary_or': UNARY_OR,
    'add': ADD, 'sub': SUB, 'mul': MUL
}


# ############# General helpers ################# #
def to_binary(value, num_bits):
    return [int(i) for i in '{:0{w}b}'.format(value, w=num_bits)][::-1]


def all_possible_inputs(num_bits):
    return pack_bool_matrix(np.array(
        [to_binary(i, num_bits) for i in range(2**num_bits)],
        dtype=np.byte))


def packed_zeros(shape):
    return pack_bool_matrix(np.zeros(shape, dtype=np.uint8))


# ############ Helpers for fixtures ############# #
def harnesses_with_property(bool_property_name):
    for name in TEST_NETWORKS:
        with open(name) as f:
            test = yaml.safe_load(f)
            if test[bool_property_name]:
                yield name


HARNESS_CACHE = dict()


def harness_to_fixture(fname, evaluator_type):
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            test = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = test

    test = deepcopy(HARNESS_CACHE[fname])

    if evaluator_type == 'standard':
        return standard_harness_to_fixture(test)
    elif evaluator_type == 'chained':
        return chained_harness_to_fixture(test)


def standard_harness_to_fixture(test):
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    samples = np.array(test['samples'], np.uint32)

    # add non-existant sub-dictionaries
    test['input matrix'] = {}
    test['output matrix'] = {}

    target = np.array(test['target matrix']['full'], dtype=np.uint8)
    activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    error = np.array(test['error matrix']['full'], dtype=np.uint8)
    inputs = activation[:, :Ni]
    test['target matrix']['full'] = target
    test['input matrix']['full'] = inputs
    test['output matrix']['full'] = activation[:, -No:]
    test['activation matrix']['full'] = activation
    test['error matrix']['full'] = error

    # generate sample versions
    target_s = target[samples]
    inputs_s = inputs[samples]
    activation_s = activation[samples]
    # add sample version of expectations to test
    test['target matrix']['sample'] = target_s
    test['input matrix']['sample'] = inputs_s
    test['output matrix']['sample'] = activation_s[:, -No:]
    test['activation matrix']['sample'] = activation_s
    test['error matrix']['sample'] = error[samples]

    Ne = inputs.shape[0]
    Ne_s = inputs_s.shape[0]
    Ne_t = Ne - Ne_s
    test['Ne'] = {'full': Ne, 'sample': Ne_s, 'test': Ne_t}

    # Test sample version
    samples_t = np.array([i for i in range(Ne_s) if i not in samples])

    target_t = target[samples_t]
    inputs_t = inputs[samples_t]
    activation_t = activation[samples_t]
    # add test version of expectations to test
    test['target matrix']['test'] = target_t
    test['input matrix']['test'] = inputs_t
    test['output matrix']['test'] = activation_t[:, -No:]
    test['activation matrix']['test'] = activation_t
    test['error matrix']['test'] = error[samples_t]

    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['network'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['network'] = BoolNetwork(gates, Ni, No)

    # add evaluators
    test['evaluator'] = {
        'sample': StandardNetworkState(test['network'],
                                       pack_bool_matrix(inputs_s),
                                       pack_bool_matrix(target_s),
                                       Ne_s),
        'full': StandardNetworkState(test['network'],
                                     pack_bool_matrix(inputs),
                                     pack_bool_matrix(target),
                                     Ne),
        'test': StandardNetworkState(test['network'],
                                     pack_bool_matrix(inputs_t),
                                     pack_bool_matrix(target_t),
                                     Ne_t)
    }

    return test


def chained_harness_to_fixture(test):
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)

    Ne_f = 2**Ni
    indices_s = np.array(test['samples'], dtype=np.uint32)
    indices_f = np.arange(Ne_f, dtype=np.uint32)
    Ne_s = indices_s.size
    Ne_t = Ne_f - Ne_s
    test['Ne'] = {'full': Ne_f, 'sample': Ne_s, 'test': Ne_t}

    op = operator_map[test['target function']]

    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['network'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['network'] = BoolNetwork(gates, Ni, No)

    if test['target function'].startswith('unary'):
        Nb = Ni
    else:
        Nb = Ni // 2

    ex_factory_s = OperatorExampleFactory(indices_s, Nb, op)
    ex_factory_f = OperatorExampleFactory(indices_f, Nb, op)
    ex_factory_t = OperatorExampleFactory(indices_s, Nb, op, Ne_f)

    generator_s = PackedExampleGenerator(ex_factory_s, No)
    generator_f = PackedExampleGenerator(ex_factory_f, No)
    generator_t = PackedExampleGenerator(ex_factory_t, No)

    window_size_s = np.random.randint(1, max(2, Ne_s // PACKED_SIZE))
    window_size_f = np.random.randint(1, max(2, Ne_f // PACKED_SIZE))
    window_size_t = np.random.randint(1, max(2, Ne_t // PACKED_SIZE))

    # add evaluators to test
    test['evaluator'] = {
        'sample': ChainedNetworkState(test['network'], generator_s, window_size_s),
        'full': ChainedNetworkState(test['network'], generator_f, window_size_f),
        'test': ChainedNetworkState(test['network'], generator_t, window_size_t),
    }

    return test


@fixture(params=['sample', 'full'])
def sample_type(request):
    return request.param


@fixture(params=['standard', 'chained'])
def evaluator_type(request):
    return request.param


@fixture(params=TEST_NETWORKS)
def state(request, evaluator_type):
    return harness_to_fixture(request.param, evaluator_type)


@fixture(params=TEST_NETWORKS)
def state_params(request):
    fname = request.param
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            harness = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = harness
    # copy to ensure harness cache remains correct
    test = deepcopy(HARNESS_CACHE[fname])

    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)

    Ne_f = 2**Ni
    indices_s = np.array(test['samples'], dtype=np.uint32)
    indices_f = np.arange(Ne_f, dtype=np.uint32)
    Ne_s = indices_s.size
    Ne_t = Ne_f - Ne_s
    test['Ne'] = {'full': Ne_f, 'sample': Ne_s, 'test': Ne_t}
    test['N'] = {'full': 0, 'sample': 0, 'test': Ne_f}

    test['operator'] = operator_map[test['target function']]

    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['network'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['network'] = BoolNetwork(gates, Ni, No)

    if test['target function'].startswith('unary'):
        test['Nb'] = Ni
    else:
        test['Nb'] = Ni // 2

    # add generated params to test
    test['indices'] = {
        'sample': indices_s,
        'full': indices_f,
        'test': indices_s
    }
    test['window_size'] = {
        'sample': np.random.randint(1, max(2, Ne_s // PACKED_SIZE)),
        'full': np.random.randint(1, max(2, Ne_f // PACKED_SIZE)),
        'test': np.random.randint(1, max(2, Ne_t // PACKED_SIZE))
    }

    return test


@fixture(params=list(harnesses_with_property('invariant under single move')))
def single_move_invariant(request):
    return harness_to_fixture(request.param, 'standard')


@fixture(params=list(harnesses_with_property('invariant under multiple moves')))
def multiple_move_invariant(request):
    return harness_to_fixture(request.param, 'standard')


MoveAndExpected = namedtuple('MoveAnExpected', ['move', 'expected'])


# ################### Functionality Testing ################### #
class TestStandard:
    @fixture(params=TEST_NETWORKS)
    def standard_state(self, request):
        return harness_to_fixture(request.param, 'standard')

    @fixture
    def single_layer_zero(self):
        instance = harness_to_fixture('boolnet/test/networks/single_layer_zero.yaml',
                                      'standard')

        instance = copy(instance)

        test_case = instance['multiple_moves_test_case']

        updated_test_case = []
        for step in test_case:
            move = step['move']
            expected = np.array(step['expected'], dtype=np.uint8)
            updated_test_case.append(MoveAndExpected(move=move, expected=expected))
        instance['multiple_moves_test_case'] = updated_test_case
        return instance

    # ################### Exception Testing ################### #
    @mark.parametrize("net, Ni, Tshape, Ne", [
        (BoolNetwork([(0, 1)], 1, 1), 1, (2**4, 1), 2**4),  # Ne != inputs.Ne
        (BoolNetwork([(0, 1)], 1, 1), 4, (2, 1), 2**4),     # Ne != target.Ne
        (BoolNetwork([(0, 1)], 1, 1), 1, (2, 4), 4),        # net.No != #targets
        (BoolNetwork([(0, 1)], 1, 1), 2, (1, 4), 4),        # net.Ni != #inputs
        (BoolNetwork([(0, 1)], 1, 1), 2, (2, 4), 4)         # both
    ])
    def test_static_construction_exceptions(self, net, Ni, Tshape, Ne):
        inp = all_possible_inputs(Ni)
        tgt = packed_zeros(Tshape)
        with raises(ValueError):
            StandardNetworkState(net, inp, tgt, Ne)

    def build_instance(self, instance_dict, sample_type, field):
        evaluator = instance_dict['evaluator'][sample_type]
        Ne = instance_dict['Ne'][sample_type]
        expected = np.array(instance_dict[field][sample_type], dtype=np.uint8)
        eval_func = lambda mat: unpack_bool_matrix(mat, Ne)
        return evaluator, expected, eval_func

    def test_input_matrix(self, standard_state, sample_type):
        evaluator, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'input matrix')
        actual = eval_func(evaluator.input_matrix)
        assert_array_equal(expected, actual)

    def test_target_matrix(self, standard_state, sample_type):
        evaluator, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'target matrix')
        actual = eval_func(evaluator.target_matrix)
        assert_array_equal(expected, actual)

    def test_output_matrix(self, standard_state, sample_type):
        evaluator, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'output matrix')
        actual = eval_func(evaluator.output_matrix)
        assert_array_equal(expected, actual)

    def test_activation_matrix(self, standard_state, sample_type):
        evaluator, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'activation matrix')
        actual = eval_func(evaluator.activation_matrix)
        assert_array_equal(expected, actual)

    def test_error_matrix(self, standard_state, sample_type):
        evaluator, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'error matrix')
        actual = eval_func(evaluator.error_matrix)
        assert_array_equal(expected, actual)

    def output_different_helper(self, instance):
        evaluator, _, eval_func = self.build_instance(instance, 'full',
                                                      'error matrix')
        net = evaluator.network
        for k in range(10):
            old_error = eval_func(evaluator.error_matrix)
            net.move_to_random_neighbour()
            assert not np.array_equal(eval_func(evaluator.error_matrix), old_error)
            net.revert_move()

    def test_single_move_output_different(self, single_move_invariant):
        self.output_different_helper(single_move_invariant)

    def test_multiple_move_output_different(self, multiple_move_invariant):
        self.output_different_helper(multiple_move_invariant)

    def test_move_with_initial_evaluation(self, single_layer_zero):
        evaluator, expected, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')

        net = evaluator.network

        actual = eval_func(evaluator.error_matrix)
        assert_array_equal(expected, actual)

        test_case = single_layer_zero['multiple_moves_test_case'][4]
        expected = test_case.expected

        net.apply_move(test_case.move)
        actual = eval_func(evaluator.error_matrix)
        assert_array_equal(expected, actual)

    def test_multiple_moves_error_matrix(self, single_layer_zero):
        evaluator, _, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')
        net = evaluator.network

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, expected in test_case:
            net.apply_move(move)
            actual = eval_func(evaluator.error_matrix)
            assert_array_equal(expected, actual)

    def test_multiple_reverts_error_matrix(self, single_layer_zero):
        evaluator, _, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')

        net = evaluator.network

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, _ in test_case:
            net.apply_move(move)

        for _, expected in reversed(test_case):
            actual = eval_func(evaluator.error_matrix)
            assert_array_equal(expected, actual)
            net.revert_move()

    def test_pre_evaluated_network(self, standard_state):
        evaluator_s, expected_s, eval_func_s = self.build_instance(
            standard_state, 'sample', 'activation matrix')
        evaluator_f, expected_f, eval_func_f = self.build_instance(
            standard_state, 'full', 'activation matrix')

        net = evaluator_s.network
        for i in range(10):
            net.move_to_random_neighbour()
            evaluator_s.evaluate()
        net.revert_all_moves()

        # check sample evaluator is still giving original results
        actual = eval_func_s(evaluator_s.activation_matrix)
        assert_array_equal(expected_s, actual)

        # check full evaluator is still giving original results
        evaluator_f.set_network(net)
        actual = eval_func_f(evaluator_f.activation_matrix)
        assert_array_equal(expected_f, actual)


class TestBoth:
    def build_from_params(self, params, eval_type, sample_type):
        if eval_type == 'standard':
            return standard_from_operator(
                network=params['network'],
                indices=params['indices'][sample_type],
                Nb=params['Nb'],
                No=params['No'],
                operator=params['operator'],
                N=params['N'][sample_type]
            )
        elif eval_type == 'chained':
            return chained_from_operator(
                network=params['network'],
                indices=params['indices'][sample_type],
                Nb=params['Nb'],
                No=params['No'],
                operator=params['operator'],
                window_size=params['window_size'][sample_type],
                N=params['N'][sample_type]
            )

    def test_from_operator_combined_attributes(self, state_params, evaluator_type, sample_type):
        state = self.build_from_params(state_params, evaluator_type, sample_type)

        Ne = state_params['Ne'][sample_type]
        assert state.Ne == Ne
        assert state.Ni == state_params['Ni']
        assert state.No == state_params['No']
        assert state.Ng == state_params['network'].Ng
        assert state.zero_mask == generate_end_mask(Ne)
        # assert state.Ne == state_params['Ne'][sample_type]

    def test_from_operator_metric_value(self, state_params, evaluator_type, metric, sample_type):
        expected = state_params['metric value'][sample_type][metric_name(metric)]
        state = self.build_from_params(state_params, evaluator_type, sample_type)
        actual = state.metric_value(metric)
        assert_array_almost_equal(expected, actual)

    def test_metric_value(self, state, metric, sample_type):
        evaluator = state['evaluator'][sample_type]
        expected = state['metric value'][sample_type][metric_name(metric)]
        actual = evaluator.metric_value(metric)
        assert_array_almost_equal(expected, actual)

    def test_multiple_metric_values_pre(self, state, sample_type):
        evaluator = state['evaluator'][sample_type]
        for metric in all_metrics():
            evaluator.add_metric(metric)
        for metric in all_metrics():
            expected = state['metric value'][sample_type][metric_name(metric)]
            actual = evaluator.metric_value(metric)
            assert_array_almost_equal(expected, actual)

    def test_multiple_metric_values_post(self, state, sample_type):
        evaluator = state['evaluator'][sample_type]
        for metric in all_metrics():
            expected = state['metric value'][sample_type][metric_name(metric)]
            actual = evaluator.metric_value(metric)
            assert_array_almost_equal(expected, actual)

import glob
import yaml
import numpy as np
import os.path
from collections import namedtuple
from pytest import fixture, yield_fixture
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from boolnet.bintools.packing import pack_bool_matrix, unpack_bool_matrix
from boolnet.bintools.packing import PACKED_SIZE_PY as PACKED_SIZE
from boolnet.bintools.metric_names import all_metrics
from boolnet.bintools.example_generator import PackedExampleGenerator, OperatorExampleFactory
from boolnet.bintools.example_generator import ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL
from boolnet.learning.networkstate import StaticNetworkState, ChainedNetworkState
import pyximport
pyximport.install()


TEST_LOCATION = 'boolnet/test/'


operator_map = {
    'zero': ZERO, 'and': AND, 'or': OR,
    'unary_and': UNARY_AND, 'unary_or': UNARY_OR,
    'add': ADD, 'sub': SUB, 'mul': MUL
}


@fixture
def test_location():
    return TEST_LOCATION


# ############ Helpers for fixtures ############# #
def harnesses_with_property(bool_property_name):
    for name in glob.glob(TEST_LOCATION + 'networks/*.yaml'):
        with open(name) as f:
            test = yaml.safe_load(f)
            if test[bool_property_name]:
                yield name


def harness_to_fixture(stream, evaluator_type):
    if evaluator_type == 'standard':
        return standard_harness_to_fixture(stream)
    elif evaluator_type == 'chained':
        return chained_harness_to_fixture(stream)
    else:
        raise ValueError('Unimplemented evaluator: ' + evaluator_type)


def standard_harness_to_fixture(stream):
    test = yaml.safe_load(stream)
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
        'sample': StaticNetworkState(test['network'],
                                     pack_bool_matrix(inputs_s),
                                     pack_bool_matrix(target_s),
                                     Ne_s),
        'full': StaticNetworkState(test['network'],
                                   pack_bool_matrix(inputs),
                                   pack_bool_matrix(target),
                                   Ne),
        'test': StaticNetworkState(test['network'],
                                   pack_bool_matrix(inputs_t),
                                   pack_bool_matrix(target_t),
                                   Ne_t)
    }

    return test


def chained_harness_to_fixture(stream):
    test = yaml.safe_load(stream)
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


# #################### Fixtures ############################ #
@fixture(params=list(all_metrics()))
def metric(request):
    return request.param


@fixture(params=glob.glob(TEST_LOCATION + '/error matrices/*.yaml'))
def error_matrix_harness(request):
    with open(request.param) as f:
        test = yaml.safe_load(f)
    folder = os.path.dirname(request.param)
    Ep = np.load(os.path.join(folder, test['name'] + '.npy'))
    E = unpack_bool_matrix(Ep, test['Ne'])
    test['packed error matrix'] = Ep
    test['unpacked error matrix'] = E
    return test


@fixture(params=['sample', 'full'])
def sample_type(request):
    return request.param


@fixture(params=['standard', 'chained'])
def evaluator_type(request):
    return request.param


@fixture(params=glob.glob(TEST_LOCATION + 'networks/*.yaml'))
def standard_state(request):
    with open(request.param) as f:
        harness = harness_to_fixture(f, 'standard')
    return harness


@fixture(params=glob.glob(TEST_LOCATION + 'networks/*.yaml'))
def state(request, evaluator_type):
    with open(request.param) as f:
        harness = harness_to_fixture(f, evaluator_type)
    return harness


@yield_fixture(params=list(harnesses_with_property(
    'invariant under single move')))
def single_move_invariant(request):
    with open(request.param) as f:
        yield harness_to_fixture(f, 'standard')


@yield_fixture(params=list(harnesses_with_property(
    'invariant under multiple moves')))
def multiple_move_invariant(request):
    with open(request.param) as f:
        yield harness_to_fixture(f, 'standard')


Move = namedtuple('Move', ['gate', 'terminal', 'source'])
MoveAndExpected = namedtuple('MoveAnExpected', ['move', 'expected'])


@fixture
def single_layer_zero():
    with open(TEST_LOCATION + 'networks/single_layer_zero.yaml') as f:
        instance = harness_to_fixture(f, 'standard')
    test_case = instance['multiple_moves_test_case']

    updated_test_case = []
    for step in test_case:
        move = Move(gate=step['move']['gate'],
                    source=step['move']['source'],
                    terminal=step['move']['terminal'])
        expected = np.array(step['expected'], dtype=np.uint8)
        updated_test_case.append(MoveAndExpected(move=move, expected=expected))
    instance['multiple_moves_test_case'] = updated_test_case
    return instance

import glob
import yaml
import numpy as np
import os.path
from collections import namedtuple
from pytest import fixture, yield_fixture
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from boolnet.bintools.packing import pack_bool_matrix, unpack_bool_matrix
from boolnet.bintools.metric_names import all_metrics
from boolnet.learning.networkstate import StaticNetworkState, ChainedNetworkState
import pyximport
pyximport.install()


TEST_LOCATION = 'boolnet/test/'


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
    if evaluator_type == 'static':
        return static_harness_to_fixture(stream)
    elif evaluator_type == 'chained':
            return static_harness_to_fixture(stream)
    else:
        raise ValueError('Unimplemented evaluator: ' + evaluator_type)


def static_harness_to_fixture(stream):
    test = yaml.safe_load(stream)
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    samples = np.array(test['samples'], np.uint32)

    # add non-existant sub-dictionaries
    test['input matrix'] = {}
    test['output matrix'] = {}

    full_target = np.array(test['target matrix']['full'], dtype=np.uint8)
    full_activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    full_error = np.array(test['error matrix']['full'], dtype=np.uint8)
    full_inputs = full_activation[:, :Ni]
    test['target matrix']['full'] = full_target
    test['input matrix']['full'] = full_inputs
    test['output matrix']['full'] = full_activation[:, -No:]
    test['activation matrix']['full'] = full_activation
    test['error matrix']['full'] = full_error

    # generate sample versions
    sample_target = full_target[samples]
    sample_inputs = full_inputs[samples]
    sample_activation = full_activation[samples]
    # add sample version of expectations to test
    test['target matrix']['sample'] = sample_target
    test['input matrix']['sample'] = sample_inputs
    test['output matrix']['sample'] = sample_activation[:, -No:]
    test['activation matrix']['sample'] = sample_activation
    test['error matrix']['sample'] = full_error[samples]

    test['Ne'] = {'full': full_activation.shape[0],
                  'sample': sample_activation.shape[0]}

    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['network'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['network'] = BoolNetwork(gates, Ni, No)
    Ne_sample = sample_inputs.shape[0]
    Ne_full = full_inputs.shape[0]
    # add evaluators to test
    test['evaluator'] = {
        'sample': StaticNetworkState(test['network'],
                                     pack_bool_matrix(sample_inputs),
                                     pack_bool_matrix(sample_target),
                                     Ne_sample),
        'full': StaticNetworkState(test['network'],
                                   pack_bool_matrix(full_inputs),
                                   pack_bool_matrix(full_target),
                                   Ne_full)
    }

    return test


def chained_harness_to_fixture(stream):
    test = yaml.safe_load(stream)
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    samples = np.array(test['samples'], np.uint32)

    # add non-existant sub-dictionaries
    test['input matrix'] = {}
    test['output matrix'] = {}

    full_target = np.array(test['target matrix']['full'], dtype=np.uint8)
    full_activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    full_error = np.array(test['error matrix']['full'], dtype=np.uint8)
    full_inputs = full_activation[:, :Ni]
    test['target matrix']['full'] = full_target
    test['input matrix']['full'] = full_inputs
    test['output matrix']['full'] = full_activation[:, -No:]
    test['activation matrix']['full'] = full_activation
    test['error matrix']['full'] = full_error

    # generate sample versions
    sample_target = full_target[samples]
    sample_inputs = full_inputs[samples]
    sample_activation = full_activation[samples]
    # add sample version of expectations to test
    test['target matrix']['sample'] = sample_target
    test['input matrix']['sample'] = sample_inputs
    test['output matrix']['sample'] = sample_activation[:, -No:]
    test['activation matrix']['sample'] = sample_activation
    test['error matrix']['sample'] = full_error[samples]

    test['Ne'] = {'full': full_activation.shape[0],
                  'sample': sample_activation.shape[0]}

    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['network'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['network'] = BoolNetwork(gates, Ni, No)
    Ne_sample = sample_inputs.shape[0]
    Ne_full = full_inputs.shape[0]
    # add evaluators to test
    test['evaluator'] = {
        'sample': StaticNetworkState(test['network'],
                                     pack_bool_matrix(sample_inputs),
                                     pack_bool_matrix(sample_target),
                                     Ne_sample),
        'full': StaticNetworkState(test['network'],
                                   pack_bool_matrix(full_inputs),
                                   pack_bool_matrix(full_target),
                                   Ne_full)
    }

    return test


# #################### Fixtures ############################ #

# @fixture(params=['static', 'chained'])
@fixture(params=['static'])
def evaluator_type(request):
    return request.param


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


@fixture(params=glob.glob(TEST_LOCATION + 'networks/*.yaml'))
def network_file_instance(request, evaluator_type):
    return request.param


@yield_fixture(params=glob.glob(TEST_LOCATION + 'networks/*.yaml'))
def any_test_network(request, evaluator_type):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_type)


@yield_fixture(params=list(harnesses_with_property(
    'invariant under single move')))
def single_move_invariant(request, evaluator_type):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_type)


@yield_fixture(params=list(harnesses_with_property(
    'invariant under multiple moves')))
def multiple_move_invariant(request, evaluator_type):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_type)


Move = namedtuple('Move', ['gate', 'terminal', 'source'])
MoveAndExpected = namedtuple('MoveAnExpected', ['move', 'expected'])


@fixture
def single_layer_zero(evaluator_type):
    with open(TEST_LOCATION + 'networks/single_layer_zero.yaml') as f:
        instance = harness_to_fixture(f, evaluator_type)
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

# content of conftest.py
import pytest
import glob
import yaml
import os
import numpy as np
from BoolNet.boolnetwork import BoolNetwork, RandomBoolNetwork
from BoolNet.packing import pack_bool_matrix
from BoolNet.metric_names import all_metrics
import pyximport
pyximport.install()
import BoolNet.networkstate


if os.path.basename(os.getcwd()) == 'BoolNet':
    TEST_LOCATION = 'test/'
else:
    TEST_LOCATION = 'BoolNet/test/'


# ############ Helpers for fixtures ############# #
def harnesses_with_property(bool_property_name):
    for name in glob.glob(TEST_LOCATION + 'networks/*.yaml'):
        with open(name) as f:
            test = yaml.safe_load(f)
            if test[bool_property_name]:
                yield name


def harness_to_fixture(stream, evaluator_class):
    test = yaml.safe_load(stream)
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    samples = np.array(test['samples'], np.uint32)

    full_target = np.array(test['target matrix']['full'], dtype=np.uint8)
    full_activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    full_error = np.array(test['error matrix']['full'], dtype=np.uint8)
    full_inputs = full_activation[:, :Ni]
    test['target matrix']['full'] = full_target
    test['activation matrix']['full'] = full_activation
    test['error matrix']['full'] = full_error
    # generate sample versions
    sample_target = full_target[samples]
    sample_inputs = full_inputs[samples]
    # add sample version of expectations to test
    test['target matrix']['sample'] = sample_target
    test['activation matrix']['sample'] = full_activation[samples]
    test['error matrix']['sample'] = full_error[samples]
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
        'sample': evaluator_class(test['network'],
                                  pack_bool_matrix(sample_inputs),
                                  pack_bool_matrix(sample_target),
                                  Ne_sample),
        'full': evaluator_class(test['network'],
                                pack_bool_matrix(full_inputs),
                                pack_bool_matrix(full_target),
                                Ne_full)
    }

    return test


# #################### Fixtures ############################ #
# @pytest.fixture(params=['static', 'dynamic'])
@pytest.fixture(params=['static'])
def evaluator_class(request):
    if request.param == 'static':
        return BoolNet.networkstate.StaticNetworkState
    else:
        return BoolNet.networkstate.DynamicNetworkSta


@pytest.fixture(params=list(all_metrics()))
def metric(request):
    return request.param


# @pytest.fixture(scope='module', autouse=True)
# def compile_BoolNetwork():
#     # compilation fixture
#     subprocess.check_call(['python3', 'setup.py', 'build_ext', '--inplace'])


@pytest.yield_fixture(params=glob.glob(TEST_LOCATION + '/error matrices/*.yaml'))
def error_matrix_harness(request):
    with open(request.param) as f:
        test = yaml.safe_load(f)
        E = np.array(test['error matrix'], dtype=int)
        test['packed error matrix'] = pack_bool_matrix(E)
        test['Ne'] = E.shape[0]
        yield test


@pytest.fixture(params=['sample', 'full'])
def network_type(request):
    return request.param


@pytest.yield_fixture(params=glob.glob(TEST_LOCATION + 'networks/*.yaml'))
def any_test_network(request, evaluator_class):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_class)


@pytest.yield_fixture(params=list(harnesses_with_property(
    'invariant under single move')))
def single_move_invariant(request, evaluator_class):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_class)


@pytest.yield_fixture(params=list(harnesses_with_property(
    'invariant under multiple moves')))
def multiple_move_invariant(request, evaluator_class):
    with open(request.param) as f:
        yield harness_to_fixture(f, evaluator_class)


@pytest.yield_fixture
def single_layer_zero(evaluator_class):
    with open(TEST_LOCATION + 'networks/single_layer_zero.yaml') as f:
        yield harness_to_fixture(f, evaluator_class)


@pytest.yield_fixture
def adder2(evaluator_class):
    with open(TEST_LOCATION + 'networks/adder2.yaml') as f:
        yield harness_to_fixture(f, evaluator_class)

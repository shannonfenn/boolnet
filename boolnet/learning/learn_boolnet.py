import time
import random
import pyximport
pyximport.install()
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from boolnet.bintools.functions import E1, ACCURACY, PER_OUTPUT, function_from_name
from boolnet.exptools.boolmapping import FileBoolMapping, OperatorBoolMapping
from boolnet.learning.networkstate import (StandardNetworkState, standard_from_operator,
                                           chained_from_operator)
import boolnet.learning.learners as learners
import boolnet.learning.optimisers as optimisers
import boolnet.exptools.fastrand as fastrand
import numpy as np
import sys
import os


OPTIMISERS = {
    'SA': optimisers.SA(),
    'LAHC': optimisers.LAHC(),
    }


LEARNERS = {
    'basic': learners.basic,
    'stratified': learners.stratified,
    'stratified kfs': learners.stratified
    }


def check_data(training_mapping, test_mapping):
    if not training_mapping.Ni:
        raise ValueError('Training inputs empty!')
    if not training_mapping.No:
        raise ValueError('Training target empty!')
    if not test_mapping.Ni:
        raise ValueError('Test inputs empty!')
    if not test_mapping.No:
        raise ValueError('Test target empty!')
    if training_mapping.Ni != test_mapping.Ni:
        raise ValueError('Training ({}) and Test ({}) Ni do not match.'.format(
            training_mapping.Ni, test_mapping.Ni))
    if training_mapping.No != test_mapping.No:
        raise ValueError('Training ({}) and Test ({}) No do not match.'.format(
            training_mapping.No, test_mapping.No))


def build_training_evaluator(network, mapping):
    if isinstance(mapping, FileBoolMapping):
        return StandardNetworkState(network, mapping.inputs, mapping.target, mapping.Ne)
    elif isinstance(mapping, OperatorBoolMapping):
        return standard_from_operator(network=network, indices=mapping.indices,
                                      Nb=mapping.Nb, No=mapping.No,
                                      operator=mapping.operator, N=mapping.N)


def build_test_evaluator(network, mapping, parameters, guiding_functions):
    if isinstance(mapping, FileBoolMapping):
        evaluator = StandardNetworkState(network, mapping.inputs, mapping.target, mapping.Ne)
    elif isinstance(mapping, OperatorBoolMapping):
        evaluator = chained_from_operator(
            network=network, indices=mapping.indices, Nb=mapping.Nb, No=mapping.No,
            operator=mapping.operator, window_size=mapping.window_size, N=mapping.N)
        # pre-add functions to avoid redundant network evaluations
        for func in guiding_functions:
            evaluator.add_function(func)
    return evaluator


def seed_rng(value):
    # seed fast random number generator using system rng (which auto seeds on module import)
    if value is not None:
        np.random.seed(value)
    else:
        random.seed()
        seed = random.randint(1, sys.maxsize)
    fastrand.seed(seed)


def build_initial_network(parameters, training_data):
    Ni, No = training_data.Ni, training_data.No

    # Create the initial connection matrix
    if 'initial_gates' in parameters['network']:
        initial_gates = np.asarray(parameters['network']['initial_gates'], dtype=np.int32)
        Ng = initial_gates.shape[0]
        # Currently only works for NAND
        if parameters['network']['node_funcs'] != 'NAND':
            raise ValueError('Given initial network only supported for node_funcs=NAND')
    else:
        Ng = parameters['network']['Ng']
        # generate random feedforward network
        initial_gates = np.empty(shape=(Ng, 2), dtype=np.int32)
        for g in range(Ng):
            initial_gates[g, :] = np.random.randint(g+Ni, size=2)
    # create the seed network
    node_funcs = parameters['network']['node_funcs']
    if node_funcs == 'random':
        # generate a random set of transfer functions
        transfer_functions = np.random.randint(16, size=Ng)
        network = RandomBoolNetwork(initial_gates, Ni, No, transfer_functions)
    elif node_funcs == 'NOR':
        # 1 is the decimal code for NOR
        transfer_functions = [1]*Ng
        network = RandomBoolNetwork(initial_gates, Ni, No, transfer_functions)
    elif node_funcs == 'NAND':
        network = BoolNetwork(initial_gates, Ni, No)
    else:
        raise ValueError('Invalid setting for \'transfer functions\': {}'.format(node_funcs))
    return network


def setup_local_dirs(parameters):
    # this ensures that the required temp directories exist in the event this
    # is executed remotely on a seperate filesystem from runexp.py
    inter_file_base = parameters['learner']['inter_file_base']
    os.makedirs(os.path.dirname(inter_file_base), exist_ok=True)


def learn_bool_net(parameters):
    start_time = time.monotonic()

    setup_local_dirs(parameters)
    seed_rng(parameters.get('seed'))

    learner_parameters = parameters['learner']
    optimiser_parameters = parameters['learner']['optimiser']

    training_data = parameters['training_mapping']
    test_data = parameters['test_mapping']
    check_data(training_data, test_data)

    initial_network = build_initial_network(parameters, training_data)

    # make evaluators for the training and test sets
    training_evaluator = build_training_evaluator(initial_network, training_data)

    learner = LEARNERS[learner_parameters['name']]
    optimiser = OPTIMISERS[optimiser_parameters['name']]

    setup_end_time = time.monotonic()

    # learn the network
    learner_result = learner(training_evaluator, learner_parameters, optimiser)

    learning_end_time = time.monotonic()

    results = build_result_map(parameters, learner_result, training_evaluator, test_data)

    end_time = time.monotonic()

    # add timing results
    results['setup_time'] = setup_end_time - start_time
    results['learning_time'] = learning_end_time - setup_end_time
    results['result_time'] = end_time - learning_end_time
    results['time'] = end_time - start_time

    return results


def build_result_map(parameters, learner_result, training_evaluator, test_data):
    learner_parameters = parameters['learner']
    optimiser_parameters = parameters['learner']['optimiser']

    guiding_function = function_from_name(optimiser_parameters['guiding_function'])

    final_network = learner_result.best_states[-1]

    training_evaluator.set_network(final_network)
    test_evaluator = build_test_evaluator(final_network, test_data, parameters,
                                          [guiding_function, E1, ACCURACY, PER_OUTPUT])

    results = {
        'Ni':                       final_network.Ni,
        'No':                       final_network.No,
        'Ng':                       final_network.Ng,
        'learner':                  learner_parameters['name'],
        'configuration_number':     parameters['configuration_number'],
        'training_set_number':      parameters['training_set_number'],
        'transfer_functions':       parameters['network']['node_funcs'],
        'iteration_for_best':       learner_result.best_iterations,
        'total_iterations':         learner_result.final_iterations,
        'training_error_simple':    training_evaluator.function_value(E1),
        'training_accuracy':        training_evaluator.function_value(ACCURACY),
        'test_error_simple':        test_evaluator.function_value(E1),
        'test_accuracy':            test_evaluator.function_value(ACCURACY),
        'final_network':            np.array(final_network.gates),
        'Ne':                       training_evaluator.Ne
        }

    # add ' kfs' on the end of the learner name in the result dict if required
    if learner_parameters.get('kfs'):
        results['learner'] += ' kfs'

    if learner_result.feature_sets:
        for bit, v in enumerate(learner_result.feature_sets):
            key = 'feature_set_target_{}'.format(bit)
            results[key] = v
    for bit, v in enumerate(training_evaluator.function_value(PER_OUTPUT)):
        key = 'training_error_target_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_evaluator.function_value(PER_OUTPUT)):
        key = 'test_error_target_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(final_network.max_node_depths()):
        key = 'max_depth_target_{}'.format(bit)
        results[key] = v
    for k, v in optimiser_parameters.items():
        results['optimiser_' + k] = v

    return results

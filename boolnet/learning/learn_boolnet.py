import time
import random
import pyximport
pyximport.install()
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from boolnet.bintools.functions import E1, ACCURACY, PER_OUTPUT, function_from_name
from boolnet.exptools.boolmapping import FileBoolMapping, OperatorBoolMapping
from boolnet.learning.learners import basic_learn, stratified_learn
from boolnet.learning.optimisers import SA, LAHC
from boolnet.learning.networkstate import (StandardNetworkState, standard_from_operator,
                                           chained_from_operator)
import boolnet.exptools.fastrand as fastrand
import numpy as np
import sys


OPTIMISERS = {
    'SA': SA(),
    'LAHC': LAHC(),
    # 'TS': TabuSearch(),
    # 'SA-VN': SA_VN(),
    # 'LAHC-VN': LAHC_VN()
    }


LEARNERS = {
    'basic': basic_learn,
    'stratified': stratified_learn,
    'stratified kfs': stratified_learn
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


def learn_bool_net(parameters):
    start_time = time.monotonic()

    # seed fast random number generator using system rng (which auto seeds on module import)
    if 'seed' in parameters:
        seed = parameters['seed']
        np.random.seed(seed)
    else:
        random.seed()
        seed = random.randint(1, sys.maxsize)
    fastrand.seed(seed)

    optimiser_name = parameters['optimiser']['name']
    learner_name = parameters['learner']['name']
    if parameters['learner'].get('kfs', False):
        learner_name += ' kfs'
    guiding_function = function_from_name(parameters['optimiser']['guiding_function'])

    training_data = parameters['training_mapping']
    test_data = parameters['test_mapping']

    check_data(training_data, test_data)

    Ni = training_data.Ni
    No = training_data.No

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
    if parameters['network']['node_funcs'] == 'random':
        # generate a random set of transfer functions
        transfer_functions = np.random.randint(16, size=Ng)
        initial_network = RandomBoolNetwork(initial_gates, Ni, No, transfer_functions)
    elif parameters['network']['node_funcs'] == 'NOR':
        # 1 is the decimal code for NOR
        transfer_functions = [1]*Ng
        initial_network = RandomBoolNetwork(initial_gates, Ni, No, transfer_functions)
    elif parameters['network']['node_funcs'] == 'NAND':
        initial_network = BoolNetwork(initial_gates, Ni, No)
    else:
        raise ValueError('Invalid setting for \'transfer functions\': {}'.format(
            parameters['network']['node_funcs']))

    # make evaluators for the training and test sets
    training_evaluator = build_training_evaluator(initial_network, training_data)

    learner = LEARNERS[learner_name]
    optimiser = OPTIMISERS[optimiser_name]

    # learn the network
    setup_time = time.monotonic()

    learner_result = learner(training_evaluator, parameters, optimiser)

    final_network = learner_result.best_states[-1]

    learning_time = time.monotonic()

    training_evaluator.clear_all_function_masks()
    training_evaluator.set_network(final_network)
    test_evaluator = build_test_evaluator(final_network, test_data, parameters,
                                          [guiding_function, E1, ACCURACY, PER_OUTPUT])

    results = {
        'Ni':                       Ni,
        'No':                       No,
        'Ng':                       Ng,
        'learner':                  learner_name,
        'configuration_number':      parameters['configuration_number'],
        'training_set_number':      parameters['training_set_number'],
        'transfer_functions':       parameters['network']['node_funcs'],
        # 'Final Network':            network_trg,
        'iteration_for_best':       learner_result.best_iterations,
        'total_iterations':         learner_result.final_iterations,
        'training_error_guiding':   training_evaluator.function_value(guiding_function),
        'training_error_simple':    training_evaluator.function_value(E1),
        'training_accuracy':        training_evaluator.function_value(ACCURACY),
        'test_error_guiding':       test_evaluator.function_value(guiding_function),
        'test_error_simple':        test_evaluator.function_value(E1),
        'test_accuracy':            test_evaluator.function_value(ACCURACY),
        'final_network':            np.array(final_network.gates),
        'Ne':                       training_evaluator.Ne,
        }

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
    for k, v in parameters['optimiser'].items():
        results['optimiser_' + k] = v

    end_time = time.monotonic()

    results['setup_time'] = setup_time - start_time
    results['learning_time'] = learning_time - setup_time
    results['result_time'] = end_time - learning_time
    results['time'] = end_time - start_time

    return results

    # if parameters['optimiser_name'] == 'anneal':
    #     parameters['finalNg'] = parameters['Ng']
    # elif parameters['optimiser_name'] == 'ganneal':

    # elif parameters['optimiser_name'] == 'tabu':
    #     raise ValueError('Tabu search not implemented yet.')

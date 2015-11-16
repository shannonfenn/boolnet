import time
import random
from boolnet.bintools.functions import (
    E1, ACCURACY, PER_OUTPUT, function_from_name)
from boolnet.exptools.boolmapping import FileBoolMapping, OperatorBoolMapping
from boolnet.learning.networkstate import (
    StandardNetworkState, standard_from_operator, chained_from_operator)
import boolnet.learning.learners as learners
import boolnet.learning.optimisers as optimisers
import boolnet.exptools.fastrand as fastrand
import numpy as np
import sys
import os


OPTIMISERS = {
    'SA': optimisers.SA(),
    'LAHC': optimisers.LAHC(),
    'HC': optimisers.HC(),
    }


LEARNERS = {
    'basic': learners.BasicLearner(),
    'stratified': learners.StratifiedLearner(),
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


def build_training_state(gates, mapping):
    if isinstance(mapping, FileBoolMapping):
        return StandardNetworkState(gates, mapping.inputs,
                                    mapping.target, mapping.Ne)
    elif isinstance(mapping, OperatorBoolMapping):
        return standard_from_operator(gates, mapping.indices, mapping.Nb,
                                      mapping.No, mapping.operator, mapping.N)


def build_test_state(gates, mapping, guiding_funcs):
    if isinstance(mapping, FileBoolMapping):
        evaluator = StandardNetworkState(gates, mapping.inputs,
                                         mapping.target, mapping.Ne)
    elif isinstance(mapping, OperatorBoolMapping):
        evaluator = chained_from_operator(
            gates, mapping.indices, mapping.Nb, mapping.No,
            mapping.operator, mapping.window_size, mapping.N)
        # pre-add functions to avoid redundant network evaluations
        for f in guiding_funcs:
            evaluator.add_function(f)
    return evaluator


def seed_rng(value):
    # seed fast random number generator using system rng which auto seeds
    # on module import
    if value is not None:
        np.random.seed(value)
    else:
        random.seed()
        seed = random.randint(1, sys.maxsize)
    fastrand.seed(seed)


def build_initial_network(parameters, training_data):
    Ni = training_data.Ni

    # Create the initial connection matrix
    if 'initial_gates' in parameters['network']:
        gates = np.asarray(parameters['network']['initial_gates'],
                           dtype=np.int32)
    else:
        Ng = parameters['network']['Ng']
        # generate random feedforward network
        gates = np.empty(shape=(Ng, 3), dtype=np.int32)
        for g in range(Ng):
            gates[g, 0] = np.random.randint(g + Ni)
            gates[g, 1] = np.random.randint(g + Ni)

        # create the seed network
        node_funcs = parameters['network']['node_funcs']
        if isinstance(node_funcs, list):
            if max(node_funcs) > 15 or min(node_funcs) < 0:
                raise ValueError('Invalid setting for \'node_funcs\': {}'.
                                 format(node_funcs))
            gates[:, 2] = np.random.choice(node_funcs, size=Ng)
        elif node_funcs == 'random':
            # generate a random set of transfer functions
            gates[:, 2] = np.random.randint(16, size=Ng)
        elif node_funcs == 'NOR':
            gates[:, 2] = 1
        elif node_funcs == 'NAND':
            gates[:, 2] = 7
        else:
            raise ValueError('Invalid setting for \'node_funcs\': {}'.
                             format(node_funcs))

    return gates


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

    gates = build_initial_network(parameters, training_data)

    # make evaluators for the training and test sets
    training_state = build_training_state(gates, training_data)

    learner = LEARNERS[learner_parameters['name']]
    optimiser = OPTIMISERS[optimiser_parameters['name']]

    setup_end_time = time.monotonic()

    # learn the network
    learner_result = learner.run(training_state, learner_parameters,
                                 optimiser)

    learning_end_time = time.monotonic()

    results = build_result_map(parameters, learner_result,
                               training_data, test_data)

    end_time = time.monotonic()

    if parameters.get('record_initial_net', False):
        results['initial_network'] = gates

    # add timing results
    results['setup_time'] = setup_end_time - start_time
    results['learning_time'] = learning_end_time - setup_end_time
    results['result_time'] = end_time - learning_end_time
    results['time'] = end_time - start_time

    return results


def build_result_map(parameters, learner_result, training_data, test_data):
    learner_parameters = parameters['learner']
    optimiser_parameters = parameters['learner']['optimiser']

    guiding_function = function_from_name(
        optimiser_parameters['guiding_function'])

    final_network = learner_result.best_states[-1]

    trg_state = build_training_state(final_network.gates, training_data)

    funcs = [guiding_function, E1, ACCURACY, PER_OUTPUT]
    test_state = build_test_state(final_network.gates, test_data, funcs)

    results = {
        'Ni':                       final_network.Ni,
        'No':                       final_network.No,
        'Ng':                       final_network.Ng,
        'learner':                  learner_parameters['name'],
        'configuration_number':     parameters['configuration_number'],
        'training_set_number':      parameters['training_set_number'],
        'training_indices':         parameters['training_indices'],
        'transfer_functions':       parameters['network']['node_funcs'],
        'iteration_for_best':       learner_result.best_iterations,
        'total_iterations':         learner_result.final_iterations,
        'training_error_simple':    trg_state.function_value(E1),
        'training_accuracy':        trg_state.function_value(ACCURACY),
        'test_error_simple':        test_state.function_value(E1),
        'test_accuracy':            test_state.function_value(ACCURACY),
        'Ne':                       trg_state.Ne
        }

    if parameters.get('record_final_net', True):
        results['final_network'] = np.array(final_network.gates)

    if parameters.get('record_intermediate_nets', False):
        for i in range(len(learner_result.best_states) - 1):
            key = 'intermediate_network_{}'.format(i)
            results[key] = learner_result.best_states[i]

    # add ' kfs' on the end of the learner name in the result dict if required
    if learner_parameters.get('kfs'):
        results['learner'] += ' kfs'

    if learner_result.feature_sets is not None:
        for strata, strata_f_sets in enumerate(learner_result.feature_sets):
            for target, v in enumerate(strata_f_sets):
                # only record FSes if they exist
                if v is not None:
                    key = 'fs_strata_{}_tgt_{}'.format(strata, target)
                    results[key] = v

    if learner_result.target_order is not None:
        results['target_order'] = learner_result.target_order

    if learner_result.restarts is not None:
        results['optimiser_restarts'] = learner_result.restarts

    for bit, v in enumerate(trg_state.function_value(PER_OUTPUT)):
        key = 'train_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_state.function_value(PER_OUTPUT)):
        key = 'test_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(final_network.max_node_depths()):
        key = 'max_depth_tgt_{}'.format(bit)
        results[key] = v
    for k, v in optimiser_parameters.items():
        results['optimiser_' + k] = v

    return results

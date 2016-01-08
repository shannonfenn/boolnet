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


# def build_initial_network(parameters):
#     gates = np.asarray(parameters['network']['initial_gates'],
#                        dtype=np.int32)
#     if gates.shape[1] != 3 or max(gates[2, :]) > 15 or min(gates[2, :]) < 0:
#         raise ValueError('Invalid initial gates: {}'.format(gates))


def build_random_network(Ng, Ni, node_funcs):
    # generate random feedforward network
    gates = np.empty(shape=(Ng, 3), dtype=np.int32)
    for g in range(Ng):
        gates[g, 0] = np.random.randint(g + Ni)
        gates[g, 1] = np.random.randint(g + Ni)
    gates[:, 2] = np.random.choice(node_funcs, size=Ng)
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

    learner_parameters['gate_generator'] = build_random_network
    learner_parameters['mapping'] = training_data

    # TODO: if mapping is OperatorBoolMapping - need to build a standard mapping
    # needs to provide: Ne, packed_input and packed_targets members
# def build_training_state(gates, mapping):
#     if isinstance(mapping, FileBoolMapping):
#         return StandardNetworkState(gates, mapping.inputs,
#                                     mapping.target, mapping.Ne)
#     elif isinstance(mapping, OperatorBoolMapping):
#         return standard_from_operator(gates, mapping.indices, mapping.Nb,
#                                       mapping.No, mapping.operator, mapping.N)

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

    # add timing results
    if parameters.get('verbose_timing'):
        results['setup_time'] = setup_end_time - start_time
        results['result_time'] = end_time - learning_end_time
        results['total_time'] = end_time - start_time
    results['learning_time'] = learning_end_time - setup_end_time

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
        'Ni':           final_network.Ni,
        'No':           final_network.No,
        'Ng':           final_network.Ng,
        'learner':      learner_parameters['name'],
        'config_num':   parameters['configuration_number'],
        'trg_set_num':  parameters['training_set_number'],
        'tfs':          parameters['network']['node_funcs'],
        'best_step':    learner_result.best_iterations,
        'steps':        learner_result.final_iterations,
        'trg_error':    trg_state.function_value(E1),
        'trg_acc':      trg_state.function_value(ACCURACY),
        'test_error':   test_state.function_value(E1),
        'test_acc':     test_state.function_value(ACCURACY),
        'Ne':           trg_state.Ne
        }

    if parameters.get('verbose_errors'):
        results['trg_err_gf'] = trg_state.function_value(guiding_function)
        results['trg_err_per'] = trg_state.function_value(PER_OUTPUT)
        results['test_err_gf'] = test_state.function_value(guiding_function)
        results['test_err_per'] = test_state.function_value(PER_OUTPUT)

    if parameters.get('record_training_indices', True):
        results['trg_indices'] = parameters['training_indices']

    if parameters.get('record_final_net', True):
        results['final_net'] = np.array(final_network.gates)

    if parameters.get('record_intermediate_nets', False):
        for i in range(len(learner_result.best_states) - 1):
            key = 'net_{}'.format(i)
            results[key] = np.array(learner_result.best_states[i].gates)

    # add ' kfs' on the end of the learner name in the result dict if required
    if learner_parameters.get('kfs'):
        results['learner'] += ' kfs'

    if learner_result.feature_sets is not None:
        for strata, strata_f_sets in enumerate(learner_result.feature_sets):
            for target, v in enumerate(strata_f_sets):
                # only record FSes if they exist
                if v is not None:
                    key = 'fs_s{}_t{}'.format(strata, target)
                    results[key] = v

    if learner_result.target_order is not None:
        results['tgt_order'] = learner_result.target_order

    if learner_result.restarts is not None:
        results['restarts'] = learner_result.restarts

    for bit, v in enumerate(trg_state.function_value(PER_OUTPUT)):
        key = 'trg_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_state.function_value(PER_OUTPUT)):
        key = 'test_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(final_network.max_node_depths()):
        key = 'max_depth_tgt_{}'.format(bit)
        results[key] = v
    for k, v in optimiser_parameters.items():
        if k == 'guiding_function':
            results[k] = v
        else:
            results['opt_' + k] = v

    return results

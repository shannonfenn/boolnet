import time
import random
from boolnet.bintools.functions import (
    E1, ACCURACY, PER_OUTPUT, function_from_name)
from boolnet.bintools.packing import partition_packed, sample_packed
from boolnet.bintools.example_generator import packed_from_operator
from boolnet.learning.networkstate import (
    StandardBNState, chained_from_operator)
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


def seed_rng(value):
    # seed fast random number generator using system rng which auto seeds
    # on module import
    if value is not None:
        np.random.seed(value)
    else:
        random.seed()
        seed = random.randint(1, sys.maxsize)
    fastrand.seed(seed)


def random_network(Ng, Ni, node_funcs):
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


def build_training_set(mapping):
    if mapping['type'] == 'raw':
        return sample_packed(mapping['matrix'], mapping['training_indices'])
    elif mapping['type'] == 'operator':
        indices = mapping['training_indices']
        operator = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        return packed_from_operator(indices, Nb, No, operator)


def learn_bool_net(parameters):
    start_time = time.monotonic()

    setup_local_dirs(parameters)
    seed_rng(parameters.get('seed'))

    learner_params = parameters['learner']
    optimiser_params = parameters['learner']['optimiser']

    learner_params['training_set'] = build_training_set(parameters['mapping'])
    learner_params['gate_generator'] = random_network

    learner = LEARNERS[learner_params['name']]
    optimiser = OPTIMISERS[optimiser_params['name']]

    setup_end_time = time.monotonic()

    # learn the network
    learner_result = learner.run(optimiser, learner_params)

    learning_end_time = time.monotonic()

    results = build_result_map(parameters, learner_result)

    end_time = time.monotonic()

    # add timing results
    if parameters.get('verbose_timing'):
        results['setup_time'] = setup_end_time - start_time
        results['result_time'] = end_time - learning_end_time
        results['total_time'] = end_time - start_time
    results['learning_time'] = learning_end_time - setup_end_time

    return results


def build_states(gates, mapping, guiding_funcs):
    if mapping['type'] == 'raw':
        M = mapping['matrix']
        indices = mapping['training_indices']
        M_trg, M_test = partition_packed(M, indices)

        I_trg, T_trg = np.split(M_trg, [M.Ni])
        I_test, T_test = np.split(M_test, [M.Ni])

        S_trg = StandardBNState(gates, I_trg, T_trg, M.Ne)
        S_test = StandardBNState(gates, I_test, T_test, M.Ne)

    elif mapping['type'] == 'operator':
        indices = mapping['training_indices']
        operator = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        window_size = mapping['window_size']

        S_trg = chained_from_operator(
            gates, indices, Nb, No, operator, window_size, exclude=False)
        S_test = chained_from_operator(
            gates, indices, Nb, No, operator, window_size, exclude=True)
        # pre-add functions to avoid redundant network evaluations
        for f in guiding_funcs:
            S_trg.add_function(f)
            S_test.add_function(f)

    return S_trg, S_test


def build_result_map(parameters, learner_result):
    learner_parameters = parameters['learner']
    optimiser_params = parameters['learner']['optimiser']

    guiding_function = function_from_name(
        optimiser_params['guiding_function'])

    final_network = learner_result.best_states[-1]
    gates = final_network.gates

    # build evaluators for training and test data
    funcs = [guiding_function, E1, ACCURACY, PER_OUTPUT]
    train_state, test_state = build_states(parameters['mapping'], gates, funcs)

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
        'trg_error':    train_state.function_value(E1),
        'trg_acc':      train_state.function_value(ACCURACY),
        'test_error':   test_state.function_value(E1),
        'test_acc':     test_state.function_value(ACCURACY),
        'Ne':           train_state.Ne
        }

    if parameters.get('verbose_errors'):
        results['trg_err_gf'] = train_state.function_value(guiding_function)
        results['trg_err_per'] = train_state.function_value(PER_OUTPUT)
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

    for bit, v in enumerate(train_state.function_value(PER_OUTPUT)):
        key = 'trg_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_state.function_value(PER_OUTPUT)):
        key = 'test_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(final_network.max_node_depths()):
        key = 'max_depth_tgt_{}'.format(bit)
        results[key] = v
    for k, v in optimiser_params.items():
        if k == 'guiding_function':
            results[k] = v
        else:
            results['opt_' + k] = v

    return results

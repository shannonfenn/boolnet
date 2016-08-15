import time
import random
import logging
import re
import numpy as np
import boolnet.bintools.functions as fn
import boolnet.bintools.packing as pk
import boolnet.bintools.example_generator as gen
import boolnet.learning.networkstate as netstate
import boolnet.learning.learners as learners
import boolnet.learning.optimisers as optimisers
import boolnet.exptools.fastrand as fastrand
import boolnet.exptools.config_filtering as cf


OPTIMISERS = {
    'SA': optimisers.SA(),
    'LAHC': optimisers.LAHC(),
    'LAHC_perc': optimisers.LAHC_perc(),
    'HC': optimisers.HC(),
    }


LEARNERS = {
    'basic': learners.BasicLearner(),
    'stratified': learners.StratifiedLearner(),
    }


def seed_rng(seed):
    # if seed is None this will use OS randomness source to generate a seed
    if seed is None:
        random.seed()
        seed = random.randint(1, 2**32)
    np.random.seed(seed)
    fastrand.seed(seed)
    return seed


def random_network(Ng, Ni, node_funcs):
    # generate random feedforward network
    gates = np.empty(shape=(Ng, 3), dtype=np.int32)
    for g in range(Ng):
        gates[g, 0] = np.random.randint(g + Ni)
        gates[g, 1] = np.random.randint(g + Ni)
    gates[:, 2] = np.random.choice(node_funcs, size=Ng)
    return gates


def build_training_set(mapping):
    if mapping['type'] == 'raw':
        return pk.sample_packed(mapping['matrix'], mapping['training_indices'])
    elif mapping['type'] == 'operator':
        indices = mapping['training_indices']
        operator = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        return gen.packed_from_operator(indices, Nb, No, operator)


def learn_bool_net(parameters):
    start_time = time.monotonic()

    seed = seed_rng(parameters['learner'].get('seed', None))
    # if no given seed then store to allow reporting in results
    parameters['learner']['seed'] = seed

    learner_params = parameters['learner']
    optimiser_params = parameters['learner']['optimiser']

    training_set = build_training_set(parameters['mapping'])
    learner_params['training_set'] = training_set
    learner_params['gate_generator'] = random_network

    # Handle flexible network size
    if str(learner_params['network']['Ng']).endswith('n'):
        n = int(str(learner_params['network']['Ng'])[:-1])
        Ng = n * training_set.No
        learner_params['network']['Ng'] = Ng

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


def build_states(mapping, gates, objectives):
    ''' objectives should be a list of (func_id, ordering, name) tuples.'''
    if mapping['type'] == 'raw':
        M = mapping['matrix']
        indices = mapping['training_indices']
        M_trg, M_test = pk.partition_packed(M, indices)

        S_trg = netstate.StandardBNState(gates, M_trg)
        S_test = netstate.StandardBNState(gates, M_test)

    elif mapping['type'] == 'operator':
        indices = mapping['training_indices']
        operator = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        window_size = mapping['window_size']

        S_trg = netstate.chained_from_operator(
            gates, indices, Nb, No, operator, window_size, exclude=False)
        S_test = netstate.chained_from_operator(
            gates, indices, Nb, No, operator, window_size, exclude=True)
    else:
        raise ValueError('Invalid mapping type: {}'.format(mapping['type']))

    # add functions to be later called by name
    for func, order, name in objectives:
        S_trg.add_function(func, order, name)
        S_test.add_function(func, order, name)

    return S_trg, S_test


def build_result_map(parameters, learner_result):

    guiding_function = fn.function_from_name(
        parameters['learner']['optimiser']['guiding_function'])

    final_network = learner_result.network
    gates = final_network.gates

    # build evaluators for training and test data
    target_order = np.array(learner_result.target_order, dtype=np.uintp)
    objective_functions = [
        (guiding_function, target_order, 'guiding'),
        (fn.E1, target_order, 'e1'),
        (fn.E1_MCC, target_order, 'e1_mcc'),
        (fn.CORRECTNESS, target_order, 'correctness'),
        (fn.PER_OUTPUT_ERROR, target_order, 'per_output_error'),
        (fn.PER_OUTPUT_MCC, target_order, 'per_output_mcc')]

    train_state, test_state = build_states(
        parameters['mapping'], gates, objective_functions)

    results = {
        'Ni':           final_network.Ni,
        'No':           final_network.No,
        'Ng':           final_network.Ng,
        'best_step':    learner_result.best_iterations,
        'steps':        learner_result.final_iterations,
        'trg_error':    train_state.function_value('e1'),
        'trg_cor':      train_state.function_value('correctness'),
        'trg_mcc':      train_state.function_value('e1_mcc'),
        'trg_err_gf':   train_state.function_value('guiding'),
        'test_error':   test_state.function_value('e1'),
        'test_cor':     test_state.function_value('correctness'),
        'test_mcc':     test_state.function_value('e1_mcc'),
        'test_err_gf':  test_state.function_value('guiding'),
        'Ne':           train_state.Ne,
        'tgt_order':    target_order
        }

    # Optional results
    if parameters.get('record_final_net', True):
        results['final_net'] = np.array(final_network.gates)

    if parameters.get('record_intermediate_nets', False):
        for i, net in enumerate(learner_result.partial_networks):
            key = 'net_{}'.format(i)
            results[key] = np.array(net.gates)

    if learner_result.feature_sets is not None:
        for strata, strata_f_sets in enumerate(learner_result.feature_sets):
            for target, fs in enumerate(strata_f_sets):
                # only record FSes if they exist
                if fs is not None:
                    key = 'fs_s{}_t{}'.format(strata, target)
                    results[key] = fs

    if learner_result.restarts is not None:
        results['restarts'] = learner_result.restarts

    # multi-part results
    for bit, v in enumerate(train_state.function_value('per_output_error')):
        key = 'trg_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_state.function_value('per_output_error')):
        key = 'test_err_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(train_state.function_value('per_output_mcc')):
        key = 'trg_mcc_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(test_state.function_value('per_output_mcc')):
        key = 'test_mcc_tgt_{}'.format(bit)
        results[key] = v
    for bit, v in enumerate(final_network.max_node_depths()):
        key = 'max_depth_tgt_{}'.format(bit)
        results[key] = v

    # handle requests to log keys
    log_keys = parameters.get('log_keys', [])

    # strip out warning flags
    log_keys_just_paths = [[k, v] for k, _, v in log_keys]

    # match dict paths to given patterns and pull out corresponding values
    passed_through_params = cf.filter_keys(parameters, log_keys_just_paths)

    # Generate warnings for reserved keys - needs to run before merging dicts
    for key, _, _ in log_keys:
        if key in results:
            logging.warning(
                'log_keys: %s ignored - reserved for results.', key)

    # Generating warnings for missing but required patterns
    for key, required, pattern in log_keys:
        # handle keys that contain insert positions
        if '{}' in key:
            checker = re.compile(key.format('.*')).fullmatch
        else:
            checker = re.compile(key).fullmatch
        # check at least one key in the result dict matches the given key
        if required and all(checker(k) is None for k in passed_through_params):
            logging.warning(('log_keys: %s is required but does not match any '
                             'path in the configuration.'), pattern)

    # merge dictionaries, giving preference to results ahead of parameters
    passed_through_params.update(results)

    return passed_through_params

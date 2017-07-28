import time
import random
import logging
import re
import numpy as np
import bitpacking.packing as pk
import boolnet.bintools.functions as fn
import boolnet.bintools.example_generator as gen
import boolnet.network.networkstate as ns
import boolnet.utils as utils
import boolnet.learners as learners
import boolnet.optimisers as optimisers
import boolnet.exptools.fastrand as fastrand
import boolnet.exptools.config_filtering as cf


OPTIMISERS = {
    'SA': optimisers.SA(),
    'LAHC': optimisers.LAHC(),
    'HC': optimisers.HC(),
    }


LEARNERS = {
    'monolithic': learners.MonolithicLearner(),
    'stratified': learners.StratifiedLearner(),
    'split': learners.SplitLearner(),
    }


def seed_rng(seed):
    # if seed is None this will use OS randomness source to generate a seed
    if seed is None:
        random.seed()
        seed = random.randint(1, 2**32)
    np.random.seed(seed)
    fastrand.seed(seed)
    return seed


def random_network(Ng, Ni, No, node_funcs):
    # generate random feedforward network
    gates = np.empty(shape=(Ng, 3), dtype=np.int32)
    for g in range(Ng):
        # don't allow connecting outputs together
        gates[g, 0] = np.random.randint(min(g, Ng - No) + Ni)
        gates[g, 1] = np.random.randint(min(g, Ng - No) + Ni)
    gates[:, 2] = np.random.choice(node_funcs, size=Ng)
    return gates


def build_training_set(mapping):
    if mapping['type'] == 'raw_split':
        trg_indices = mapping.get('training_indices', None)
        if trg_indices is not None:
            return utils.sample_packed(mapping['training_set'], trg_indices)
        else:
            return mapping['training_set']

    elif mapping['type'] == 'raw_unsplit':
        return utils.sample_packed(mapping['matrix'],
                                   mapping['training_indices'])

    elif mapping['type'] == 'operator':
        indices = mapping['training_indices']
        operator = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        targets = mapping['targets']
        return gen.packed_from_operator(indices, Nb, No, operator, targets)


def add_noise(mapping, rate):
    Ne, No = mapping.Ne, mapping.No

    num_flips = int(np.round(mapping.Ne * rate))

    if num_flips == 0:
        # rather than complicating code to handle 0 flips
        return 0
    effective_rate = num_flips / Ne

    noise_mask = np.zeros((Ne, No), dtype=np.uint8)
    noise_mask[-num_flips:, :] = 1

    for i in range(No):
        np.random.shuffle(noise_mask[:, i])
    noise_mask = pk.packmat(noise_mask)

    # XOR will flip all bits where the noise mask is '1'
    mapping[-No:, :] = np.bitwise_xor(mapping[-No:, :], noise_mask).astype(mapping.dtype)

    return effective_rate


def learn_bool_net(parameters, verbose=False):
    start_time = time.monotonic()

    seed = seed_rng(parameters['learner'].get('seed', None))
    # if no given seed then store to allow reporting in results
    parameters['learner']['seed'] = seed

    learner_params = parameters['learner']
    optimiser_params = parameters['learner']['optimiser']

    training_set = build_training_set(parameters['mapping'])

    if 'add_noise' in parameters['data']:
        rate = add_noise(training_set, parameters['data']['add_noise'])
        parameters['actual_noise'] = rate

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
    learner_result = learner.run(optimiser, learner_params, verbose)

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
    if mapping['type'] == 'raw_split':
        trg_indices = mapping.get('training_indices', None)
        test_indices = mapping.get('test_indices', None)
        M_trg = mapping['training_set']
        M_test = mapping['test_set']
        if trg_indices is not None:
            M_trg = utils.sample_packed(M_trg, mapping['training_indices'])
        if test_indices is not None:
            M_test = utils.sample_packed(M_test, mapping['test_indices'])
        S_trg = ns.BNState(gates, M_trg)
        S_test = ns.BNState(gates, M_test)

    elif mapping['type'] == 'raw_unsplit':
        M = mapping['matrix']
        trg_indices = mapping['training_indices']
        test_indices = mapping['test_indices']
        if test_indices is None:
            M_trg, M_test = utils.partition_packed(M, trg_indices)
        else:
            M_trg = utils.sample_packed(M, trg_indices)
            M_test = utils.sample_packed(M, test_indices)
        S_trg = ns.BNState(gates, M_trg)
        S_test = ns.BNState(gates, M_test)

    elif mapping['type'] == 'operator':
        trg_indices = mapping['training_indices']
        test_indices = mapping['test_indices']
        if test_indices is None:
            test_indices = trg_indices
            exclude = True
        else:
            exclude = False
        op = mapping['operator']
        Nb = mapping['Nb']
        No = mapping['No']
        tgts = mapping['targets']
        M_trg = gen.packed_from_operator(trg_indices, Nb, No, op, tgts)
        M_test = gen.packed_from_operator(test_indices, Nb, No, op, tgts,
                                          exclude)
        S_trg = ns.BNState(gates, M_trg)
        S_test = ns.BNState(gates, M_test)

    else:
        raise ValueError('Invalid mapping type: {}'.format(mapping['type']))

    # add functions to be later called by name
    for func, name, params in objectives:
        S_trg.add_function(func, name, params=params)
        S_test.add_function(func, name, params=params)

    return S_trg, S_test


def build_result_map(parameters, learner_result):

    guiding_function = fn.function_from_name(
        parameters['learner']['optimiser']['guiding_function'])
    guiding_function_params = parameters['learner']['optimiser'].get(
            'guiding_function_parameters', {})

    final_network = learner_result.network
    gates = final_network.gates

    # build evaluators for training and test data
    objective_functions = [
        (guiding_function, 'guiding', guiding_function_params),
        (fn.E1, 'e1', {}),
        (fn.E1_MCC, 'e1_mcc', {}),
        (fn.CORRECTNESS, 'correctness', {}),
        (fn.PER_OUTPUT_ERROR, 'per_output_error', {}),
        (fn.PER_OUTPUT_MCC, 'per_output_mcc', {})]

    train_state, test_state = build_states(
        parameters['mapping'], gates, objective_functions)

    # Optional results
    if parameters.get('record_final_net', True):
        learner_result.extra['final_net'] = np.array(gates)

    if 'partial_networks' in learner_result.extra:
        if parameters.get('record_intermediate_nets', False):
            for i, net in enumerate(learner_result.partial_networks):
                key = 'net_{}'.format(i)
                learner_result.extra[key] = np.array(net.gates)
        learner_result.extra.pop('partial_networks')

    if 'feature_sets' in learner_result.extra:
        F = learner_result.extra['feature_sets']
        for strata, strata_f_sets in enumerate(F):
            for target, fs in enumerate(strata_f_sets):
                # only record FSes if they exist
                if fs is not None:
                    key = 'fs_s{}_t{}'.format(strata, target)
                    learner_result.extra[key] = fs
        learner_result.extra.pop('feature_sets')

    results = {
        'Ni':           final_network.Ni,
        'No':           final_network.No,
        'Ng':           final_network.Ng,
        'trg_err':    train_state.function_value('e1'),
        'trg_cor':      train_state.function_value('correctness'),
        'trg_mcc':      train_state.function_value('e1_mcc'),
        'trg_err_gf':   train_state.function_value('guiding'),
        'test_err':   test_state.function_value('e1'),
        'test_cor':     test_state.function_value('correctness'),
        'test_mcc':     test_state.function_value('e1_mcc'),
        'test_err_gf':  test_state.function_value('guiding'),
        'Ne':           train_state.Ne,
        'tgt_order':    np.array(learner_result.target_order, dtype=np.uintp),
        }
    results.update(learner_result.extra)

    if 'actual_noise' in parameters:
        results['actual_noise'] = parameters['actual_noise']

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
    # for bit, v in enumerate(final_network.max_node_depths()):
    #     key = 'max_depth_tgt_{}'.format(bit)
    #     results[key] = v

    if 'notes' in parameters:
        results['notes'] = ''.join(parameters['notes'].values())

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
                'log_keys: %s ignored - reserved key.', key)

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

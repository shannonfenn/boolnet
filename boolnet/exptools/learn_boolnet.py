import time
import random
import logging
import re
import numpy as np
import bitpacking.packing as pk
import boolnet.bintools.functions as fn
import boolnet.bintools.biterror as be
import boolnet.bintools.example_generator as gen
import boolnet.network.networkstate as ns
import boolnet.utils as utils
import boolnet.learners.monolithic as monolithic
import boolnet.learners.stratified as stratified
import boolnet.learners.stratified_multipar as stratified_multipar
import boolnet.learners.split as split
import boolnet.learners.classifierchain as classifierchain
import boolnet.learners.classifierchain_plus as classifierchain_plus
import boolnet.optimisers as optimisers
import boolnet.exptools.fastrand as fastrand
import boolnet.exptools.config_filtering as cf


OPTIMISERS = {
    'SA': optimisers.SA,
    'LAHC': optimisers.LAHC,
    'HC': optimisers.HC,
    }


LEARNERS = {
    'monolithic': monolithic.Learner(),
    'stratified': stratified.Learner(),
    'stratmultipar': stratified_multipar.Learner(),
    'split': split.Learner(),
    'classifierchain': classifierchain.Learner(),
    'classifierchain_plus': classifierchain_plus.Learner(),
    }


def seed_rng(seed):
    # if seed is None this will use OS randomness source to generate a seed
    if seed is None:
        random.seed()
        seed = random.randint(1, 2**32)
    random.seed(seed)
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


def load_dataset(fname, targets):
    with np.load(fname) as ds:
        M = utils.PackedMatrix(ds['matrix'], ds['Ne'], ds['Ni'])
    if targets is not None:
        Y = M[M.Ni:, :]
        Y = Y[targets, :]
        M = utils.PackedMatrix(np.vstack((M[:M.Ni, :], Y)), M.Ne, M.Ni)
    return M


def convert_file_datasets(parameters):
    mapping = parameters['mapping']
    if mapping['type'] == 'file_split':
        targets = mapping.get('targets', None)
        Mp_trg = load_dataset(mapping['trg_file'], targets)
        Mp_test = load_dataset(mapping['test_file'], targets)

        parameters['mapping']['type'] = 'raw_split'
        parameters['mapping']['training_set'] = Mp_trg
        parameters['mapping']['test_set'] = Mp_test

    elif mapping['type'] == 'file_unsplit':
        targets = mapping.get('targets', None)
        Mp = load_dataset(mapping['file'], targets)
        parameters['mapping']['type'] = 'raw_unsplit'
        parameters['mapping']['matrix'] = Mp


def convert_target_orders(parameters, No):
    order = parameters['target_order']
    if order == 'lsb':
        parameters['target_order'] = np.arange(No, dtype=np.uintp)
    elif order == 'msb':
        parameters['target_order'] = np.arange(No, dtype=np.uintp)[::-1]
    elif order == 'random':
        parameters['target_order'] = np.random.permutation(No).astype(np.uintp)
    elif order == 'auto':
        parameters['target_order'] = None
    else:
        parameters['target_order'] = np.array(order, dtype=np.uintp)


def fn_value_stop_criterion(func_id, evaluator, limit=None):
    if limit is None:
        limit = fn.optimum(func_id)
    if fn.is_minimiser(func_id):
        return lambda state, _: state.function_value(evaluator) <= limit
    else:
        return lambda state, _: state.function_value(evaluator) >= limit


def guiding_fn_value_stop_criterion(func_id, limit=None):
    if limit is None:
        limit = fn.optimum(func_id)
    if fn.is_minimiser(func_id):
        return lambda _, guiding_value: guiding_value <= limit
    else:
        return lambda _, guiding_value: guiding_value >= limit


def initialise_optimiser(params, Ne, No):
    params = dict(params)
    # handle guiding function
    gf_name = params['guiding_function']
    gf_id = fn.function_from_name(gf_name)
    if gf_id not in fn.scalar_functions():
        raise ValueError('Invalid guiding function: {}'.format(gf_name))
    gf_params = params.get('guiding_function_parameters', {})
    gf_evaluator = be.EVALUATORS[gf_id](Ne, No, **gf_params)
    # update
    params['minimise'] = fn.is_minimiser(gf_id)
    params['guiding_function'] = lambda x: x.function_value(gf_evaluator)

    # Handle stopping condition (may be user specified)
    # defaults to the guiding function optima
    condition = params.get('stopping_condition', ['guiding', None])
    sf_name = condition[0]
    sf_limit = condition[1]
    if sf_name != 'guiding':
        sf_id = fn.function_from_name(sf_name)
        sf_params = condition[2] if len(condition) > 2 else {}
        sf_evaluator = be.EVALUATORS[sf_id](Ne, No, **sf_params)
        stop_functor = fn_value_stop_criterion(sf_id, sf_evaluator, sf_limit)
    else:
        stop_functor = guiding_fn_value_stop_criterion(gf_id, sf_limit)
    # update
    params['stopping_condition'] = stop_functor

    # variable max iterations
    if str(params.get('max_iterations', '')).endswith('n'):
        n = int(str(params['max_iterations'])[:-1])
        max_it = n * No
        params['max_iterations'] = max_it

    name = params.pop('name')
    return OPTIMISERS[name](**params)


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

    convert_file_datasets(parameters)

    training_set = build_training_set(parameters['mapping'])

    learner_params = parameters['learner']
    convert_target_orders(learner_params, training_set.No)

    if 'add_noise' in parameters['data']:
        rate = add_noise(training_set, parameters['data']['add_noise'])
        parameters['actual_noise'] = rate

    learner_params['training_set'] = training_set

    # prepare model generator
    node_funcs = parameters['learner']['network']['node_funcs']
    if not all(f in range(16) for f in node_funcs):
        raise ValueError('\'node_funcs\' must come from [0, 15]: {}'.
                         format(node_funcs))
    learner_params['model_generator'] = lambda Ng, Ni, No: random_network(
        Ng, Ni, No, node_funcs)

    # Handle flexible network size
    if str(learner_params['network']['Ng']).endswith('n'):
        n = int(str(learner_params['network']['Ng'])[:-1])
        Ng = n * training_set.No
        learner_params['network']['Ng'] = Ng

    record = build_parameter_record(parameters)

    optimiser = initialise_optimiser(parameters['optimiser'], training_set.Ne, training_set.No)
    learner = LEARNERS[learner_params['name']]

    # learn the network
    setup_end_time = time.monotonic()
    learner_result = learner.run(optimiser, learner_params, verbose)
    learning_end_time = time.monotonic()

    record.update(build_result_record(parameters, learner_result))
    end_time = time.monotonic()

    # record timing
    if parameters.get('verbose_timing'):
        record['setup_time'] = setup_end_time - start_time
        record['result_time'] = end_time - learning_end_time
        record['total_time'] = end_time - start_time
    record['learning_time'] = learning_end_time - setup_end_time

    return record


def build_states(mapping, gates):
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

    return S_trg, S_test


def build_parameter_record(parameters):
    record = {}
    if 'actual_noise' in parameters:
        record['actual_noise'] = parameters['actual_noise']

    for k, v in parameters.items():
        if re.match(r'notes.*', k):
            record[k] = v

    # handle requests to log keys
    log_keys = parameters.get('log_keys', [])

    # strip out warning flags
    log_keys_just_paths = [[k, v] for k, _, v in log_keys]

    # match dict paths to given patterns and pull out corresponding values
    passed_through_params = cf.filter_keys(parameters, log_keys_just_paths)

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

    record.update(passed_through_params)
    return record


def build_result_record(parameters, learner_result):
    gf_id = fn.function_from_name(
        parameters['optimiser']['guiding_function'])
    gf_params = parameters['optimiser'].get(
            'guiding_function_parameters', {})

    final_network = learner_result['network']
    gates = np.asarray(final_network.gates)

    train_state, test_state = build_states(parameters['mapping'], gates)

    # build evaluators for training and test data
    eval_guiding_train = be.EVALUATORS[gf_id](train_state.Ne, train_state.No, **gf_params)
    eval_guiding_test = be.EVALUATORS[gf_id](test_state.Ne, test_state.No, **gf_params)
    eval_e1_train = be.EVALUATORS[fn.E1](train_state.Ne, train_state.No)
    eval_e1_test = be.EVALUATORS[fn.E1](test_state.Ne, test_state.No)
    eval_macro_mcc_train = be.EVALUATORS[fn.MACRO_MCC](train_state.Ne, train_state.No)
    eval_macro_mcc_test = be.EVALUATORS[fn.MACRO_MCC](test_state.Ne, test_state.No)
    eval_correctness_train = be.EVALUATORS[fn.CORRECTNESS](train_state.Ne, train_state.No)
    eval_correctness_test = be.EVALUATORS[fn.CORRECTNESS](test_state.Ne, test_state.No)
    eval_per_output_err_train = be.EVALUATORS[fn.PER_OUTPUT_ERROR](train_state.Ne, train_state.No)
    eval_per_output_err_test = be.EVALUATORS[fn.PER_OUTPUT_ERROR](test_state.Ne, test_state.No)
    eval_per_output_mcc_train = be.EVALUATORS[fn.PER_OUTPUT_MCC](train_state.Ne, train_state.No)
    eval_per_output_mcc_test = be.EVALUATORS[fn.PER_OUTPUT_MCC](test_state.Ne, test_state.No)

    # Optional results
    if parameters.get('record_final_net', True):
        learner_result['extra']['final_net'] = gates

    partial_nets = learner_result['extra'].pop('partial_networks', None)
    if partial_nets and parameters.get('record_intermediate_nets', False):
        partial_nets = [net.gates.tolist() for net in partial_nets]
        learner_result['extra']['partial_networks'] = partial_nets

    record = {
        'Ni':           final_network.Ni,
        'No':           final_network.No,
        'Ng':           final_network.Ng,
        'Ne':           train_state.Ne,
        'tgt_order':    learner_result['target_order'],
        # training set metrics
        'trg_err':      train_state.function_value(eval_e1_train),
        'trg_cor':      train_state.function_value(eval_correctness_train),
        'trg_mcc':      train_state.function_value(eval_macro_mcc_train),
        'trg_err_gf':   train_state.function_value(eval_guiding_train),
        'trg_errs':     np.asarray(train_state.function_value(eval_per_output_err_train)),
        'trg_mccs':     np.asarray(train_state.function_value(eval_per_output_mcc_train)),
        # test set metrics
        'test_err':     test_state.function_value(eval_e1_test),
        'test_cor':     test_state.function_value(eval_correctness_test),
        'test_mcc':     test_state.function_value(eval_macro_mcc_test),
        'test_err_gf':  test_state.function_value(eval_guiding_test),
        'test_errs':    np.asarray(test_state.function_value(eval_per_output_err_test)),
        'test_mccs':    np.asarray(test_state.function_value(eval_per_output_mcc_test)),
        }
    record.update(learner_result['extra'])

    return record

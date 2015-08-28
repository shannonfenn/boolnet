from copy import copy
from collections import namedtuple
from itertools import product
import sys
import logging
import pyximport
import numpy as np
pyximport.install()
from boolnet.bintools.functions import PER_OUTPUT, function_from_name
from boolnet.bintools.packing import unpack_bool_matrix, unpack_bool_vector
import boolnet.learning.kfs as kfs


Result = namedtuple('Result', [
    'best_states', 'best_iterations', 'final_iterations', 'target_order', 'feature_sets'])


def per_target_error_end_condition(bit):
    return lambda ev, _: ev.function_value(PER_OUTPUT)[bit] <= 0


def guiding_error_end_condition():
    return lambda _, error: error <= 0


def strata_boundaries(network):
    # this allocates gate boundaries linearly, that it the first
    # target is allocate the first Ng/No gates and so on.
    No, Ng = network.No, network.Ng
    upper_bounds = np.linspace((Ng-No)/No, Ng-No, No).astype(int).tolist()
    return [0] + upper_bounds


def check_parameters(parameters):
    problem_funcs = ['e{} {}'.format(i, j) for i, j in product([4, 5, 6, 7], ['msb', 'lsb'])]

    function_name = parameters['optimiser']['guiding_function']
    if function_name in problem_funcs:
        logging.error(('use of %s guiding function may result in poor performance due to non-zero '
                       'errors in earlier bits.'), function_name)
        print('WARNING POTENTIALLY ERROR PRONE METRIC FOR STRATIFIED LEARNING!', file=sys.stderr)


def handle_single_FS(feature_set, evaluator, bit, target, changeable_gate):
    # When we have a 1FS we have already learned the target, or its inverse,
    # simply map this feature to the output
    activation_matrix = evaluator.activation_matrix
    network = evaluator.network
    Ng = network.Ng
    No = network.No
    feature = feature_set[0]
    if activation_matrix[feature][0] == target[bit][0]:
        # we have the target perfectly, in this case place a double
        # inverter chain (we could take the inputs from a gate if it was one
        # but in the event the feature is an input this is not possible)
        network.apply_move({'gate': changeable_gate, 'terminal': 0, 'new_source': feature})
        network.apply_move({'gate': changeable_gate, 'terminal': 1, 'new_source': feature})
        network.apply_move({'gate': Ng - No + bit, 'terminal': 0, 'new_source': changeable_gate})
        network.apply_move({'gate': Ng - No + bit, 'terminal': 1, 'new_source': changeable_gate})
    else:
        # we have the target's inverse, since a NAND gate can act as an
        # inverter we can connect the output gate directly to the feature
        network.apply_move({'gate': Ng - No + bit, 'terminal': 0, 'new_source': feature})
        network.apply_move({'gate': Ng - No + bit, 'terminal': 1, 'new_source': feature})


def build_mask(network, lower_bound, upper_bound, target_index, feature_set=None):
    # IDEA: May later use entropy or something similar. For now
    #       this uses the union of all minimal feature sets the
    #       connection encoded (+Ni) values for the changeable
    #       gates - or else we could only have single layer
    #       networks at each stage!
    Ni = network.Ni
    Ng = network.Ng
    No = network.No
    # give list of gates which can have their inputs reconnected
    changeable = np.zeros(Ng, dtype=np.uint8)
    sourceable = np.zeros(Ng+Ni, dtype=np.uint8)

    changeable[lower_bound:upper_bound] = 1
    changeable[Ng - No + target_index] = 1

    # and a list of inputs (including gate outputs) which those
    # gates can be connected to
    if feature_set is None:
        # include all modifiable and previous connections
        sourceable[:upper_bound+Ni] = 1
    else:
        # union of all feature sets
        sourceable[feature_set.flat] = 1
        # and the range of modifiable gates (with Ni added)
        sourceable[lower_bound+Ni:upper_bound+Ni] = 1

    # include all previous final outputs
    sourceable[Ng-No+Ni:Ng-No+Ni+target_index] = 1

    # logging.info('kfs result: %s', feature_set)
    # logging.info('prior outputs: %s', range(Ng - No + Ni, Ng - No + Ni + target_index))
    # logging.info('Changeable: %s', changeable)
    # logging.info('Sourceable: %s', sourceable)

    return sourceable, changeable


def get_feature_set(evaluator, parameters, target, boundaries, all_inputs):
    activation_matrix = np.asarray(evaluator.activation_matrix)
    Ni = evaluator.network.Ni
    # generate input to minFS solver
    upper_gate = boundaries[target]
    if all_inputs or target == 0:
        input_feature_indices = np.arange(upper_gate+Ni)
    else:
        lower_gate = boundaries[target - 1]
        input_feature_indices = np.hstack((np.arange(Ni), np.arange(lower_gate+Ni, upper_gate+Ni)))

    kfs_matrix = activation_matrix[input_feature_indices, :]
    # target feature for this bit
    kfs_target = evaluator.target_matrix[target, :]

    logging.info('\t(examples, features): {}'.format(evaluator.Ne, kfs_matrix.shape))

    file_name_base = parameters['inter_file_base'] + str(target)

    kfs_matrix = unpack_bool_matrix(kfs_matrix, evaluator.Ne)
    kfs_target = unpack_bool_vector(kfs_target, evaluator.Ne)

    options = parameters.get('fabcpp_options')
    # use external solver for minFS
    feature_sets = kfs.minimal_feature_sets(kfs_matrix, kfs_target, file_name_base, options)

    # for now only one feature set should exist
    if feature_sets.shape[0] > 1:
        raise ValueError('More than one feature set returned.')

    return input_feature_indices[feature_sets][0]


def prepare_state(state, parameters, gate_boundaries, target_matrix, target, feature_set_results):
    # options
    use_kfs_masking = parameters.get('kfs', False)
    log_all_feature_sets = parameters.get('log_all_feature_sets', False)

    network = state.network
    num_targets, _ = target_matrix.shape
    lower_bound = gate_boundaries[target]
    upper_bound = gate_boundaries[target + 1]

    if use_kfs_masking:
        one_layer_kfs = parameters.get('one_layer_kfs', False)
        # find a set of min feature sets for the next target
        fs = get_feature_set(state, parameters, target, gate_boundaries, not one_layer_kfs)

        # keep a log of the feature sets found at each iteration
        if log_all_feature_sets:
            fs_list = [fs]
            for t in range(target+1, num_targets):
                fs_list.append(get_feature_set(state, parameters, lower_bound, t))
            feature_set_results.append(fs_list)
        else:
            feature_set_results.append(fs)

        # check for 1-FS, if we have a 1FS we have already learnt the
        # target, or its inverse, so simply map this feature to the output
        if fs.size == 1:
            handle_single_FS(fs, state, target, target_matrix, lower_bound)
            return False

        sourceable, changeable = build_mask(network, lower_bound, upper_bound, target, fs)
    else:
        sourceable, changeable = build_mask(network, lower_bound, upper_bound, target)

    # apply the mask to the network
    network.set_mask(sourceable, changeable)

    # TODO: Fix it so that the gate_edit boundaries aren't leaving gates unused
    #       in the event that the learner finishes before exhausting all gates

    # Reinitialise the next range of gates to be optimised according to the mask
    network.reconnect_masked_range()

    return True


def stratified(state, parameters, optimiser):
    num_targets = state.network.No
    auto_target = parameters.get('auto_target')

    check_parameters(parameters)

    # For logging network as each new bit is learnt
    best_states = [None] * num_targets
    best_iterations = [-1] * num_targets
    final_iterations = [-1] * num_targets
    # For recording the order in which targets where learned
    target_order_array = [-1] * num_targets

    # allocate gate pools for each bit, to avoid the problem of prematurely using all the gates
    gate_boundaries = strata_boundaries(state.network)
    # the initial kfs input is just the set of all network inputs
    target_matrix = np.array(state.target_matrix)

    fs_results = []

    for i in range(num_targets):
        if auto_target:
            raise NotImplemented
        else:
            target = i
            optimisation_required = prepare_state(
                state, parameters, gate_boundaries, target_matrix, target, fs_results)

        target_order_array[i] = target

        if optimisation_required:
            optimised_network, best_it, final_it = learn_single_target(
                state, parameters['optimiser'], optimiser, target)

            # record result
            best_states[target] = copy(optimised_network)
            best_iterations[target] = best_it
            final_iterations[target] = final_it
        else:
            best_states[target] = copy(optimised_network)

    return Result(best_states, best_iterations, final_iterations, target_order_array, fs_results)


def learn_single_target(state, opt_params, optimiser, target):
    opt_params = copy(opt_params)
    guiding_func_id = function_from_name(opt_params['guiding_function'])
    # generate an end condition based on the current target
    end_condition = per_target_error_end_condition(target)

    # build new guiding function for only next target if required
    if guiding_func_id == PER_OUTPUT:
        guiding_func = lambda x: x.function_value(guiding_func_id)[target]
    else:
        guiding_func = lambda x: x.function_value(guiding_func_id)

    opt_params['guiding_function'] = guiding_func

    # run the optimiser
    return optimiser.run(state, opt_params, end_condition)


def basic(evaluator, parameters, optimiser):
    ''' This just learns by using the given optimiser and guiding function.'''
    opt_params = copy(parameters['optimiser'])
    end_condition = guiding_error_end_condition()

    guiding_func_id = function_from_name(opt_params['guiding_function'])
    opt_params['guiding_function'] = lambda evaluator: evaluator.function_value(guiding_func_id)

    results = optimiser.run(evaluator, opt_params, end_condition)
    best_state, best_it, final_it = results

    return Result([best_state], [best_it], [final_it], None, None)

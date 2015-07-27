from copy import copy
from collections import namedtuple
import numpy as np
import logging
import pyximport
pyximport.install()
from boolnet.bintools.metrics import PER_OUTPUT
from boolnet.bintools.packing import unpack_bool_matrix, unpack_bool_vector
import boolnet.learning.kfs as kfs


Result = namedtuple('Result', [
    'best_states', 'best_iterations', 'final_iterations', 'feature_sets'])


def per_target_error_end_condition(bit):
    return lambda ev, _: ev.metric_value(PER_OUTPUT)[bit] <= 0


def guiding_error_end_condition():
    return lambda _, error: error <= 0


def strata_boundaries(network):
    # this allocates gate boundaries linearly, that it the first
    # target is allocate the first Ng/No gates and so on.
    No = network.No
    Ng = network.Ng
    upper_bounds = np.linspace((Ng-No)/No, Ng-No, No).astype(int).tolist()
    lower_bounds = [0] + upper_bounds[:-1]
    return lower_bounds, upper_bounds


def check_parameters(parameters):
    problem_metrics = ['e4 msb', 'e4 lsb', 'e5 msb', 'e5 lsb',
                       'e6 msb', 'e6 lsb', 'e7 msb', 'e7 lsb']

    metric_name = parameters['optimiser']['metric']
    if metric_name in problem_metrics:
        logging.error(('use of %s metric may result in poor performance '
                       'due to masking of earlier bits'), metric_name)
        # print('WARNING POTENTIALLY ERROR PRONE METRIC FOR KFS!',
        #       file=sys.stderr)


def handle_single_FS(feature_set, evaluator, bit, target):
    # When we have a 1FS we have already learned the target, or its inverse,
    # simply map this feature to the output
    activation_matrix = evaluator.activation_matrix
    network = evaluator.network
    Ni = network.Ni
    Ng = network.Ng
    No = network.No
    gate = feature_set[0] - Ni
    if activation_matrix[gate][0] == target[bit][0]:
        # we have learnt the target perfectly, in this case place the
        # current outputs inputs as the inputs to the 1FS gate
        sources = network.gates[gate]
        for i in [0, 1]:
            network.apply_move({'gate': Ng - No + bit, 'terminal': i, 'new_source': sources[i]})
    else:
        # we have learnt the target inverse, since a NAND gate can act
        # as an inverter we can connect the output gate directly to this one
        for i in [0, 1]:
            network.apply_move({'gate': Ng - No + bit, 'terminal': i, 'new_source': gate})


def get_mask(network, lower_bound, upper_bound, target_index, feature_set=None):
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


def get_feature_set(evaluator, parameters, lower_bound, bit):
    activation_matrix = evaluator.activation_matrix
    Ni = evaluator.network.Ni
    # generate input to minFS solver
    kfs_matrix = activation_matrix[:(lower_bound+Ni), :]
    # target feature for this bit
    kfs_target = evaluator.target_matrix[bit, :]

    logging.info('\t(examples, features): {}'.format(evaluator.Ne, kfs_matrix.shape))

    file_name_base = parameters['inter_file_base'] + str(bit)

    kfs_matrix = unpack_bool_matrix(kfs_matrix, evaluator.Ne)
    kfs_target = unpack_bool_vector(kfs_target, evaluator.Ne)

    # use external solver for minFS
    feature_sets = kfs.minimal_feature_sets(kfs_matrix, kfs_target, file_name_base)

    # for now only one feature set should exist
    if feature_sets.shape[0] > 1:
        raise ValueError('More than one feature set returned.')

    return feature_sets[0]


def stratified_learn(evaluator, parameters, optimiser,
                     use_kfs_masking=False, log_all_feature_sets=False):
    network = evaluator.network
    num_targets = network.No

    # For logging network as each new bit is learnt
    best_states = [None] * num_targets
    best_iterations = [-1] * num_targets
    final_iterations = [-1] * num_targets
    optimiser_parameters = parameters['optimiser']

    # allocate gate pools for each bit, to avoid the problem of prematurely using all the gates
    lower_bounds, upper_bounds = strata_boundaries(network)
    # the initial kfs input is just the set of all network inputs
    target_matrix = np.array(evaluator.target_matrix)

    feature_set_results = []
    # ################## Start kfs learning ################## #

    for tgt, L_bnd, U_bnd in zip(range(num_targets), lower_bounds, upper_bounds):
        if use_kfs_masking:
            # find a set of min feature sets for the next target
            feature_set = get_feature_set(evaluator, parameters, L_bnd, tgt)

            # keep a log of the feature sets found at each iteration
            if log_all_feature_sets:
                fs_list = [feature_set]
                for t in range(tgt+1, num_targets):
                    fs_list.append(get_feature_set(evaluator, parameters, L_bnd, t))
                feature_set_results.append(fs_list)
            else:
                feature_set_results.append(feature_set)

            # check for 1-FS
            if feature_set.size == 1:
                # When we have a 1FS we have already learned the target,
                # or its inverse, simply map this feature to the output
                handle_single_FS(feature_set, evaluator, tgt, target_matrix)
                continue

            sourceable, changeable = get_mask(network, L_bnd, U_bnd, tgt, feature_set)
        else:
            sourceable, changeable = get_mask(network, L_bnd, U_bnd, tgt)

        # apply the mask to the network
        network.set_mask(sourceable, changeable)
        # generate an end condition based on the current target
        end_condition = per_target_error_end_condition(tgt)

        # TODO: Fix it so that the gate_edit boundaries aren't
        #       leaving gates unused in the event that the learner
        #       finishes before exhausting all gates

        # Reinitialise the next range of gates to be optimised
        # according to the mask
        network.reconnect_masked_range()

        # run the optimiser
        results = optimiser.run(evaluator, optimiser_parameters, end_condition)
        # unpack results
        optimised_network, best_it, final_it = results

        # logging.info('''Error per output (sample): %s
        #                 best iteration: %d
        #                 final iteration: %d''',
        #              evaluator.metric_value(PER_OUTPUT),
        #              best_it, final_it)

        # record result
        best_states[tgt] = copy(optimised_network)
        best_iterations[tgt] = best_it
        final_iterations[tgt] = final_it

    # ################## End kfs learning ################## #

    logging.info('Best states: %s', best_states)

    # TODO: Maybe we should return the best state out of the
    #       states according to the provided guiding metric?

    if feature_set_results:
        return Result(best_states, best_iterations, final_iterations, feature_set_results)
    else:
        return Result(best_states, best_iterations, final_iterations, None)


def basic_learn(evaluator, parameters, optimiser):
    ''' This just learns by using the given optimiser and guiding metric.'''
    opt_params = parameters['optimiser']
    end_condition = guiding_error_end_condition()

    results = optimiser.run(evaluator, opt_params, end_condition)
    best_state, best_it, final_it = results

    return Result([best_state], [best_it], [final_it], None)

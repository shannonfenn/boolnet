from copy import copy
from collections import namedtuple
from itertools import product
import logging
import numpy as np
from boolnet.bintools.functions import PER_OUTPUT, function_from_name
from boolnet.bintools.packing import unpack_bool_matrix, unpack_bool_vector
import boolnet.learning.kfs as kfs


LearnerResult = namedtuple('LearnerResult', [
    'best_states', 'best_iterations', 'final_iterations',
    'target_order', 'feature_sets', 'restarts'])


def per_target_error_stop_criterion(bit):
    return lambda ev, _: ev.function_value(PER_OUTPUT)[bit] <= 0


def guiding_error_stop_criterion():
    return lambda _, error: error <= 0


def build_mask(state, lower_bound, upper_bound, target, feature_set=None):
    # IDEA: May later use entropy or something similar. For now
    #       this uses the union of all minimal feature sets the
    #       connection encoded (+Ni) values for the changeable
    #       gates - or else we could only have single layer
    #       networks at each stage!
    Ni = state.Ni
    Ng = state.Ng
    No = state.No
    # give list of gates which can have their inputs reconnected
    changeable = np.zeros(Ng, dtype=np.uint8)
    sourceable = np.zeros(Ng+Ni, dtype=np.uint8)

    changeable[lower_bound:upper_bound] = 1
    changeable[Ng - No + target] = 1

    # and a list of inputs (including gates) which those gates can be connected to
    if feature_set is None:
        # all modifiable and previous connections
        sourceable[:upper_bound+Ni] = 1
    else:
        sourceable[feature_set] = 1
        # and the range of modifiable gates (with Ni added)
        sourceable[lower_bound+Ni:upper_bound+Ni] = 1

    # include all previous final outputs
    # first_output = Ng-No+Ni
    # sourceable[first_output:first_output+target] = 1

    return sourceable, changeable


class BasicLearner:
    def _check_parameters(self, parameters):
        pass

    def _setup(self, parameters, state, optimiser):
        self._check_parameters(parameters)
        self.opt_params = copy(parameters['optimiser'])
        guiding_func_name = self.opt_params['guiding_function']
        self.guiding_func_id = function_from_name(guiding_func_name)
        self.opt_params['guiding_function'] = lambda x: x.function_value(self.guiding_func_id)
        self.optimiser = optimiser

    def _optimise(self, state):
        ''' This just learns by using the given optimiser and guiding function.'''
        return self.optimiser.run(state, self.opt_params)

    def run(self, state, parameters, optimiser):
        self._setup(parameters, state, optimiser)
        self.opt_params['stopping_criterion'] = guiding_error_stop_criterion()
        best_state, best_it, final_it, restarts = self._optimise(state)
        return LearnerResult([best_state], [best_it], [final_it], None, None, restarts)


class StratifiedLearner(BasicLearner):
    def _setup(self, parameters, state, optimiser):
        super()._setup(parameters, state, optimiser)
        self.auto_target = parameters.get('auto_target')
        self.use_kfs_masking = parameters.get('kfs')
        self.log_all_feature_sets = parameters.get('log_all_feature_sets')
        self.one_layer_kfs = parameters.get('one_layer_kfs')
        self.file_name_base = parameters['inter_file_base']
        self.fabcpp_opts = parameters.get('fabcpp_options')
        self.keep_files = parameters.get('keep_files', False)

        self.num_targets = state.network.No
        self.learned_targets = []
        self.feature_sets = np.empty((self.num_targets, self.num_targets), dtype=list)

        self.gate_boundaries = np.linspace(
            0, state.Ng - state.No, state.No+1, dtype=int)

        self.target_matrix = np.array(state.target_matrix, copy=True)

    def _check_parameters(self, parameters):
        problem_funcs = ['e{} {}'.format(i, j) for i, j in product([4, 5, 6, 7], ['msb', 'lsb'])]
        function_name = parameters['optimiser']['guiding_function']
        if function_name in problem_funcs:
            logging.error(('use of %s guiding function may result in poor performance due to '
                           'non-zero errors in earlier bits.'), function_name)

    def _determine_next_target(self, state):
        not_learned = list(set(range(self.num_targets)).difference(self.learned_targets))
        if self.auto_target:
            # raise NotImplemented
            strata = len(self.learned_targets)
            self._record_feature_sets(state, not_learned)
            feature_sets_sizes = [len(l) for l in self.feature_sets[strata][not_learned]]
            indirect_simplest_target = np.argmin(feature_sets_sizes)
            return not_learned[indirect_simplest_target]
        else:
            return min(not_learned)

    def _record_feature_sets(self, state, targets):
        # keep a log of the feature sets found at each iteration
        strata = len(self.learned_targets)
        for t in targets:
            self.feature_sets[strata, t] = self._get_single_fs(state, t, not self.one_layer_kfs)

    def _get_single_fs(self, state, target, all_strata):
        activation_matrix = np.asarray(state.activation_matrix)
        Ni = state.Ni
        strata = len(self.learned_targets)
        # generate input to minFS solver
        upper_source = self.gate_boundaries[strata] + Ni
        if all_strata or strata == 0:
            input_feature_indices = np.arange(upper_source)
        else:
            lower_source = self.gate_boundaries[strata - 1] + Ni
            input_feature_indices = np.hstack((
                np.arange(Ni), np.arange(lower_source, upper_source)))

        # input features for this target
        kfs_matrix = activation_matrix[input_feature_indices, :]
        kfs_matrix = unpack_bool_matrix(kfs_matrix, state.Ne)
        # target feature for this target
        kfs_target = self.target_matrix[target, :]
        kfs_target = unpack_bool_vector(kfs_target, state.Ne)

        kfs_filename = self.file_name_base + '{}_{}'.format(strata, target)

        # check if the target is constant 1 or 0 and if so do not run minFS
        if np.all(kfs_target) or not np.any(kfs_target):
            logging.warning('Constant target: {}'.format(target))
            # we cannot discriminate features so simply return all inputs
            return np.arange(Ni)
        else:
            # use external solver for minFS
            minfs = kfs.minimum_feature_set(kfs_matrix, kfs_target, kfs_filename, self.fabcpp_opts, self.keep_files)
            return input_feature_indices[minfs]

    def _apply_mask(self, state, target):
        # which layer we are up to
        strata = len(self.learned_targets)

        lower_bound = self.gate_boundaries[strata]
        upper_bound = self.gate_boundaries[strata + 1]

        if self.use_kfs_masking:
            if self.feature_sets[strata, target] is None:
                # find a set of min feature sets for the next target
                self._record_feature_sets(state, [target])

            fs = self.feature_sets[strata, target]
            # check for 1-FS, if we have a 1FS we have already learnt the
            # target, or its inverse, so simply map this feature to the output
            if fs.size == 1:
                self._handle_single_FS(fs[0], state, target)
                return False
        else:
            fs = None

        sourceable, changeable = build_mask(state, lower_bound, upper_bound, target, fs)

        # apply the mask to the network
        state.set_mask(sourceable, changeable)

        # TODO: Fix it so that the gate_edit boundaries aren't leaving gates unused
        #       in the event that the learner finishes before exhausting all gates

        # Reinitialise the next range of gates to be optimised according to the mask
        state.reconnect_masked_range()

        return True

    def _handle_single_FS(self, feature, state, target):
        # When we have a 1FS we have already learned the target, or its inverse,
        # simply map this feature to the output
        strata = len(self.learned_targets)
        useable_gate = self.gate_boundaries[strata]

        activation_matrix = state.activation_matrix
        target_gate = state.Ng - state.No + target
        if activation_matrix[feature][0] == self.target_matrix[target][0]:
            # we have the target perfectly, in this case place a double
            # inverter chain (we could take the inputs from a gate if it was one
            # but in the event the feature is an input this is not possible)
            state.apply_move({'gate': useable_gate, 'terminal': 0, 'new_source': feature})
            state.apply_move({'gate': useable_gate, 'terminal': 1, 'new_source': feature})
            useable_gate_input = useable_gate + state.Ni
            state.apply_move({'gate': target_gate, 'terminal': 0, 'new_source': useable_gate_input})
            state.apply_move({'gate': target_gate, 'terminal': 1, 'new_source': useable_gate_input})
        else:
            # we have the target's inverse, since a NAND gate can act as an
            # inverter we can connect the output gate directly to the feature
            state.apply_move({'gate': target_gate, 'terminal': 0, 'new_source': feature})
            state.apply_move({'gate': target_gate, 'terminal': 1, 'new_source': feature})

    def _learn_target(self, state, target):
        # build new guiding function for only next target if required
        if self.guiding_func_id == PER_OUTPUT:
            guiding_func = lambda x: x.function_value(self.guiding_func_id)[target]
            self.opt_params['guiding_function'] = guiding_func

        # generate an end condition based on the current target
        self.opt_params['stopping_criterion'] = per_target_error_stop_criterion(target)

        # run the optimiser
        return self._optimise(state)

    def run(self, state, parameters, optimiser):
        self._setup(parameters, state, optimiser)
        No = state.No

        best_states = [None] * No
        best_iterations = [-1] * No
        final_iterations = [-1] * No
        restarts = [-1] * No

        for i in range(No):
            # determine target
            target = self._determine_next_target(state)
            # apply mask
            optimisation_required = self._apply_mask(state, target)
            # optimise
            if optimisation_required:
                opt_results = self._learn_target(state, target)
                # record result
                best_states[i] = copy(opt_results[0])
                best_iterations[i] = opt_results[1]
                final_iterations[i] = opt_results[2]
                restarts[i] = opt_results[3]
            else:
                # record result
                best_states[i] = copy(state.network)
            self.learned_targets.append(target)

        return LearnerResult(best_states=best_states,
                             best_iterations=best_iterations,
                             final_iterations=final_iterations,
                             target_order=self.learned_targets,
                             feature_sets=self.feature_sets,
                             restarts=restarts)

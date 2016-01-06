from copy import copy
from collections import namedtuple
import logging
import numpy as np
from boolnet.bintools.functions import PER_OUTPUT, function_from_name
from boolnet.bintools.packing import unpack_bool_matrix, unpack_bool_vector
from boolnet.learning.networkstate import StandardNetworkState
import boolnet.learning.kfs as mfs


LearnerResult = namedtuple('LearnerResult', [
    'best_states', 'best_errors', 'best_iterations', 'final_iterations',
    'target_order', 'feature_sets', 'restarts'])


def per_tgt_err_stop_criterion(bit):
    return lambda ev, _: ev.function_value(PER_OUTPUT)[bit] <= 0


def guiding_func_stop_criterion():
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

    # and a list of inputs (inc gates) which they can source from
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

    def _setup(self, parameters, optimiser):
        self._check_parameters(parameters)
        self.opt_params = copy(parameters['optimiser'])
        guiding_func_name = self.opt_params['guiding_function']
        self.guiding_func_id = function_from_name(guiding_func_name)
        self.opt_params['guiding_function'] = (
            lambda x: x.function_value(self.guiding_func_id))
        self.optimiser = optimiser

    def _optimise(self, state):
        ''' Just learns by using the given optimiser and guiding function.'''
        return self.optimiser.run(state, self.opt_params)

    def run(self, state, parameters, optimiser):
        self._setup(parameters, state, optimiser)
        self.opt_params['stopping_criterion'] = guiding_func_stop_criterion()
        opt_result = self._optimise(state)
        return LearnerResult(best_states=[opt_result.state],
                             best_errors=[opt_result.error],
                             best_iterations=[opt_result.best_iteration],
                             final_iterations=[opt_result.iteration],
                             target_order=None,
                             feature_sets=None,
                             restarts=[opt_result.restarts])


class StratifiedLearner(BasicLearner):

    def _setup(self, parameters, optimiser):
        super()._setup(parameters, optimiser)
        # Required
        self.mfs_fname = parameters['inter_file_base']
        self.remaining_budget = parameters['network']['Ng']
        self.node_funcs = parameters['network']['node_funcs']
        self.gate_generator = parameters['gate_generator']
        # Optional
        self.auto_target = parameters.get('auto_target', False)
        self.use_mfs_selection = parameters.get('kfs', False)
        self.fabcpp_opts = parameters.get('fabcpp_options', False)
        self.keep_files = parameters.get('keep_files', False)
        # Instance
        mapping = parameters['mapping']
        self.Ne = mapping.Ne()
        self.input_matrix = mapping.packed_input()
        self.target_matrix = mapping.packed_targets()

        self.num_targets = self.target_matrix.shape[0]
        self.learned_targets = []
        self.feature_sets = np.empty((self.num_targets, self.num_targets),
                                     dtype=list)

    def _check_parameters(self, parameters):
        problem_funcs = ['PER_OUTPUT']
        function_name = parameters['optimiser']['guiding_function']
        if function_name in problem_funcs:
            raise ValueError('Invalid guiding function: {}.'
                             .format(function_name))

    def _determine_next_target(self, strata, inputs):
        all_targets = set(range(self.num_targets))
        not_learned = list(all_targets.difference(self.learned_targets))
        if self.auto_target:
            # find minFS for all unlearned targets
            self._record_feature_sets(inputs, not_learned)
            feature_sets_sizes = [len(l) for l in
                                  self.feature_sets[strata][not_learned]]
            indirect_simplest_target = np.argmin(feature_sets_sizes)
            return not_learned[indirect_simplest_target]
        elif self.use_mfs_selection:
            target = min(not_learned)
            # find minFS for just the next target
            self._record_feature_sets(inputs, [target])
            return target
        else:
            return min(not_learned)

    def _record_feature_sets(self, inputs, targets):
        # keep a log of the feature sets found at each iteration
        strata = len(self.learned_targets)
        for t in targets:
            fs = self._get_fs(inputs, t)
            self.feature_sets[strata, t] = fs

    def _get_fs(self, inputs, target_index, strata):
        # input to minFS solver
        mfs_matrix = unpack_bool_matrix(inputs, self.Ne)
        # target feature
        mfs_target = self.target_matrix[target_index, :]
        mfs_target = unpack_bool_vector(mfs_target, self.Ne)

        mfs_filename = self.mfs_fname + '{}_{}'.format(strata, target_index)

        # check if the target is constant 1 or 0 and if so do not run minFS
        if np.all(mfs_target) or not np.any(mfs_target):
            logging.warning('Constant target: {}'.format(target_index))
            # we cannot discriminate features so simply return all inputs
            return np.arange(inputs.shape[0])
        else:
            # use external solver for minFS
            minfs = mfs.minimum_feature_set(
                mfs_matrix, mfs_target, mfs_filename,
                self.fabcpp_opts, self.keep_files)
            return minfs

    def construct_partial_state(self, strata, target_index, inputs):
        # determine next budget
        size = self.remaining_budget // (self.No - strata)
        target = self.target_matrix[target_index]

        if self.use_mfs_selection:
            # subsample the inputs for below
            fs = self.feature_sets[strata, target_index]
            inputs = inputs[fs]
            # check for 1-FS, if we have a 1FS we have already learnt the
            # target, or its inverse, so simply map this feature to the output
            # NOTE: does not working with generic node function list since it
            #       relies on the presence of the NAND gate
            # if fs.size == 1:
            #     # have the below function build a network with 1 or 2
            #     # gates and return it
            #     gates = self.handle_single_FS(inputs[0], target)
            #     self.remaining_budget -= gates.shape[0]
            #     return StandardNetworkState(gates, inputs, target, self.Ne)

        gates = self.gate_generator(size, inputs.shape[0], self.node_funcs)
        self.remaining_budget -= size
        return StandardNetworkState(gates, inputs, target, self.Ne)

    def insert_network(self, base, new, strata, target_index):
        # simple: build up a map for all sources, for sources after the
        # original minfs input just have them as + Ni_offset (whatever that is,
        # might including old.Ng) and for the former ones use the fs mapping

        if self.use_mfs_selection:
            # build a map for replacing sources
            new_input_map = self.feature_sets[strata][target_index]
            offset = base.Ni + base.Ng - base.No - new.Ni
            new_gate_map = list(range(offset,
                                      offset + new.Ng - new.No))
            new_output_map = list(range(offset + new.Ng + base.No - new.No,
                                        offset + new.Ng + base.No))
            sources_map = new_input_map + new_gate_map + new_output_map

            # vectorised function for applying the map
            def mapper(entry):
                return sources_map[entry]
            mapper = np.vectorize(mapper)

            # apply to all but the last column (it is for the transfer
            # functions) and the last No rows (since they ) of the gate matrix
            remapped_new_gates = np.hstack(mapper(new.gates[:-new.No, :-1]),
                                           new.gates[:, -1])

            accumulated_gates = np.vstack((base.gates[:-base.No, :],
                                           remapped_new_gates[:-new.No, :],
                                           base.gates[-base.No:, :],
                                           remapped_new_gates[-new.No:, :]))
        else:
            # The gates can be tacked on since new.Ni = base.Ni + base.Ng
            accumulated_gates = np.vstack((base.gates[:-base.No, :],
                                           new.gates[:-new.No, :],
                                           base.gates[-base.No:, :],
                                           new.gates[-new.No:, :]))

        new_target = np.vstack((base.target_matrix, new.target_matrix))
        return StandardNetworkState(accumulated_gates, self.input_matrix,
                                    new_target)

    def reorder_network(self, state):
        # all non-output gates are left alone, and the output gates are
        # reordered by the inverse permutation of "learned_targets"
        new_gate_order = np.concatenate((
            np.arange(state.Ng - state.No),
            np.argsort(self.learned_targets).tolist()))
        new_gates = state.gates[new_gate_order]
        return StandardNetworkState(new_gates, self.input_matrix,
                                    self.target_matrix)

    def run(self, state, parameters, optimiser):

        # setup accumulated network
        # loop:
        #   make partial network
        #   optimise it
        #   hook into accumulated network
        # reorganise outputs

        self._setup(parameters, optimiser)
        No = state.No

        opt_results = []

        inputs = np.array(self.input_matrix)

        # make a state with Ng = No = 0 and set the inp mat = self.input_matrix
        accumulated_network = StandardNetworkState(
            np.empty((0, 3)), inputs, np.empty((0, inputs.shape[1])), self.Ne)

        for i in range(No):
            # determine target
            target_index = self._determine_next_target(state)

            state = self.construct_partial_state(i, target_index, inputs)

            # generate an end condition based on the current target
            criterion = guiding_func_stop_criterion(target_index)
            self.opt_params['stopping_criterion'] = criterion

            # optimise
            partial_result = self._optimise(state)
            # record result
            opt_results.append(partial_result)

            # build up final network by inserting this partial result
            accumulated_network = self.insert_network(
                accumulated_network, partial_result.state, i, target_index)

            # don't include output gates as possible inputs
            # new Ni = Ng - No + Ni
            inputs = np.array(accumulated_network.activation_matrix[:-i, :])

            self.learned_targets.append(target_index)

        # reorder the outputs to match the supplied target order
        # NOTE: This is why output gates are not included as possible inputs
        opt_results[-1].state = self.reorder_network(opt_results[-1].state)

        return LearnerResult(
            best_states=[r.state for r in opt_results],
            best_errors=[r.error for r in opt_results],
            best_iterations=[r.best_iteration for r in opt_results],
            final_iterations=[r.iteration for r in opt_results],
            target_order=self.learned_targets,
            feature_sets=self.feature_sets,
            restarts=[r.restarts for r in opt_results])

    # def handle_single_FS(self, feature, target):
    #     # When we have a 1FS we have already learned the target,
    #     # ot its inverse, simply map this feature to the output
    #     match = np.logical_and(feature, target)
    #     if np.all(match):
    #         # we have the target perfectly, in this case place a double
    #         # inverter chain (since the output must be a NAND which negates)
    #         return np.array([[0, 0, 7], [1, 1, 7]])
    #     elif not np.any(match):
    #         # we have the target's inverse, since a NAND gate can act as an
    #         # inverter we can connect the output gate directly to the feature
    #         return np.array([[0, 0, 7]])
    #     else:
    #         raise ValueError('1FS handling triggered with invalid feature:\n'
    #                          'feature:\n{}\ntarget:\n{}'
    #                          .format(feature, target))
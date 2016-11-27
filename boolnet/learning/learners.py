from copy import copy
from collections import namedtuple
import numpy as np
import boolnet.bintools.functions as fn
from boolnet.bintools.packing import (
    unpack_bool_matrix, unpack_bool_vector, BitPackedMatrix)
from boolnet.learning.networkstate import StandardBNState
import boolnet.learning.feature_selection as mfs


LearnerResult = namedtuple('LearnerResult', [
    'network', 'partial_networks', 'best_errors', 'best_iterations',
    'final_iterations', 'target_order', 'feature_sets', 'restarts'])


def guiding_func_stop_criterion(func_id, limit=None):
    if limit is None:
        limit = fn.optimum(func_id)
    if fn.is_minimiser(func_id):
        return lambda _, error: error <= limit
    else:
        return lambda _, error: error >= limit


def inverse_permutation(permutation):
    inverse = np.zeros_like(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


class BasicLearner:
    def _setup(self, optimiser, parameters):
        # Gate generation
        self.gate_generator = parameters['gate_generator']
        self.node_funcs = parameters['network']['node_funcs']
        self.budget = parameters['network']['Ng']
        # Instance
        self.problem_matrix = parameters['training_set']
        self.Ni = self.problem_matrix.Ni
        self.No = self.problem_matrix.No
        self.input_matrix, self.target_matrix = np.split(
            self.problem_matrix, [self.Ni])
        self.Ne = self.problem_matrix.Ne
        # Optimiser
        self.optimiser = optimiser
        self.opt_params = copy(parameters['optimiser'])
        gf_name = self.opt_params['guiding_function']
        self.guiding_func_id = fn.function_from_name(gf_name)
        self.opt_params['minimise'] = fn.is_minimiser(self.guiding_func_id)

        # convert shorthands for target order
        if parameters['target_order'] == 'lsb':
            self.target_order = np.arange(self.No, dtype=np.uintp)
        elif parameters['target_order'] == 'msb':
            self.target_order = np.arange(self.No, dtype=np.uintp)[::-1]
        elif parameters['target_order'] == 'auto':
            self.target_order = None
            # this key is only required if auto-targetting
            self.mfs_method = parameters['minfs_selection_method']
        else:
            self.target_order = np.array(parameters['target_order'],
                                         dtype=np.uintp)
        # Optional minfs solver time limit
        self.time_limit = parameters.get('minfs_time_limit', None)
        self.minfs_solver = parameters.get('minfs_solver', 'cplex')

        # add functors for evaluating the guiding func and stop criteria
        self.gf_eval_name = 'guiding'
        self.opt_params['guiding_function'] = lambda x: x.function_value(
            self.gf_eval_name)
        # Check if user supplied cutoff point
        limit = parameters.get('stopping_error', None)
        self.opt_params['stopping_criterion'] = guiding_func_stop_criterion(
            self.guiding_func_id, limit)

        if self.guiding_func_id not in fn.scalar_functions():
            raise ValueError('Invalid guiding function: {}'.format(gf_name))
        if max(self.node_funcs) > 15 or min(self.node_funcs) < 0:
            raise ValueError('\'node_funcs\' must come from [0, 15]: {}'.
                             format(self.node_funcs))

    def order_from_rank(self, ranks):
        ''' Converts a ranking with ties into an ordering,
            breaking ties with uniform probability.'''
        order = []
        ranks, counts = np.unique(ranks, return_counts=True)
        for rank, count in zip(ranks, counts):
            order.extend((np.random.permutation(count) + rank).tolist())
        return np.array(order, dtype=np.uintp)

    def run(self, optimiser, parameters):
        self._setup(optimiser, parameters)

        if self.target_order is None:
            # determine the target order by ranking feature sets
            mfs_features = unpack_bool_matrix(self.input_matrix, self.Ne)
            mfs_targets = unpack_bool_matrix(self.target_matrix, self.Ne)

            # use external solver for minFS
            rank, feature_sets = mfs.ranked_feature_sets(
                mfs_features, mfs_targets, self.mfs_method,
                timelimit=self.time_limit, solver=self.minfs_solver)

            # randomly pick from top ranked targets
            self.target_order = self.order_from_rank(rank)

        # build the network state
        gates = self.gate_generator(self.budget, self.Ni, self.node_funcs)
        state = StandardBNState(gates, self.problem_matrix)
        # add the guiding function to be evaluated
        state.add_function(self.guiding_func_id, self.target_order,
                           self.gf_eval_name)

        # run the optimiser
        opt_result = self.optimiser.run(state, self.opt_params)

        return LearnerResult(
            network=opt_result.representation,
            best_errors=[opt_result.error],
            best_iterations=[opt_result.best_iteration],
            final_iterations=[opt_result.iteration],
            restarts=[opt_result.restarts],
            target_order=self.target_order,
            feature_sets=None,
            partial_networks=[])


class StratifiedLearner(BasicLearner):

    def _setup(self, optimiser, parameters):
        super()._setup(optimiser, parameters)
        # Required
        self.auto_target = (parameters['target_order'] == 'auto')
        # Optional
        self.use_minfs_selection = parameters.get('minfs_masking', False)
        self.mfs_method = parameters.get('minfs_selection_method', None)
        # Initialise
        self.No, _ = self.target_matrix.shape
        self.remaining_budget = self.budget
        self.learned_targets = []
        self.feature_sets = np.empty((self.No, self.No),
                                     dtype=list)

    def determine_next_target(self, strata, inputs):
        if self.auto_target:
            # get unlearned targets
            all_targets = np.arange(self.No, dtype=int)
            not_learned = np.setdiff1d(all_targets, self.learned_targets)

            # unpack inputs to minFS solver
            mfs_features = unpack_bool_matrix(inputs, self.Ne)
            mfs_targets = unpack_bool_matrix(
                self.target_matrix[not_learned, :], self.Ne)

            # use external solver for minFS
            if strata > 0 and False:  # disabled for now
                prior_solns = self.feature_sets[strata-1, not_learned]
            else:
                prior_solns = [None]*len(not_learned)

            rank, feature_sets = mfs.ranked_feature_sets(
                mfs_features, mfs_targets, self.mfs_method, prior_solns,
                timelimit=self.time_limit, solver=self.minfs_solver)

            # replace empty feature sets
            for i in range(len(feature_sets)):
                if len(feature_sets[i]) == 0:
                    feature_sets[i] = list(range(inputs.shape[0]))

            self.feature_sets[strata, not_learned] = feature_sets

            # randomly pick from top ranked targets
            return not_learned[np.random.choice(np.where(rank == 0)[0])]
        else:
            # get next target from given ordering
            t = self.target_order[strata]
            if self.use_minfs_selection:
                # unpack inputs to minFS solver
                mfs_features = unpack_bool_matrix(inputs, self.Ne)
                mfs_target = unpack_bool_vector(self.target_matrix[t], self.Ne)
                fs, _ = mfs.best_feature_set(
                    mfs_features, mfs_target, self.mfs_method,
                    timelimit=self.time_limit, solver=self.minfs_solver)
                if len(fs) == 0:
                    fs = list(range(inputs.shape[0]))

                self.feature_sets[strata, t] = fs
            return t

    def make_partial_instance(self, strata, target_index, inputs):
        if self.use_minfs_selection:
            # subsample the inputs for below
            fs = self.feature_sets[strata, target_index]
            inputs = inputs[fs]

        target = self.target_matrix[target_index]

        return BitPackedMatrix(np.vstack((inputs, target)),
                               Ne=self.Ne, Ni=inputs.shape[0])

    def next_gates(self, strata, problem_matrix):
        size = self.remaining_budget // (self.No - strata)
        self.remaining_budget -= size
        return self.gate_generator(size, problem_matrix.Ni, self.node_funcs)

    def join_networks(self, base, new, strata, target_index):
        # simple: build up a map for all sources, for sources after the
        # original minfs input just have them as + Ni_offset (whatever that is,
        # might including old.Ng) and for the former ones use the fs mapping

        if self.use_minfs_selection:
            # build a map for replacing sources
            new_input_map = self.feature_sets[strata][target_index]
            # difference in input sizes plus # of non-output base gates
            # offset = base.Ni - new.Ni + base.Ng - base.No
            offset = base.Ni + base.Ng - base.No
            new_gate_map = list(range(offset,
                                      offset + new.Ng - new.No))
            new_output_map = list(range(offset + new.Ng - new.No + base.No,
                                        offset + new.Ng + base.No))
            sources_map = new_input_map + new_gate_map + new_output_map

            # apply to all but the last column (transfer functions) of the gate
            # matrix. Use numpy array: cython memoryview slicing is broken
            remapped_new_gates = np.array(new.gates)
            for gate in remapped_new_gates:
                for i in range(gate.size - 1):
                    gate[i] = sources_map[gate[i]]

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

        new_problem_matrix = BitPackedMatrix(
            np.vstack((self.input_matrix, new_target)),
            Ne=self.Ne, Ni=self.Ni)

        return StandardBNState(accumulated_gates, new_problem_matrix)

    def reorder_network_outputs(self, network):
        # all non-output gates are left alone, and the output gates are
        # reordered by the inverse permutation of "learned_targets"
        No = network.No
        new_out_order = inverse_permutation(self.learned_targets)
        new_gates = np.array(network.gates)
        new_gates[-No:, :] = new_gates[-No:, :][new_out_order]
        network.set_gates(new_gates)

    def run(self, optimiser, parameters):
        # setup accumulated network
        # loop:
        #   make partial network
        #   optimise it
        #   hook into accumulated network
        # reorganise outputs

        self._setup(optimiser, parameters)

        opt_results = []

        inputs = self.input_matrix.copy()

        # make a state with Ng = No = 0 and set the inp mat = self.input_matrix
        accumulated_network = StandardBNState(np.empty((0, 3)), inputs)

        for i in range(self.No):

            # determine next target index
            target = self.determine_next_target(i, inputs)

            partial_instance = self.make_partial_instance(i, target, inputs)

            # build state to be optimised
            gates = self.next_gates(i, partial_instance)
            state = StandardBNState(gates, partial_instance)
            # add the guiding function to be evaluated
            state.add_function(self.guiding_func_id,
                               np.arange(state.No, dtype=np.uintp),
                               self.gf_eval_name)

            # optimise
            partial_result = self.optimiser.run(state, self.opt_params)
            # record result
            opt_results.append(partial_result)

            # build up final network by inserting this partial result
            result_state = StandardBNState(
                partial_result.representation.gates, partial_instance)
            accumulated_network = self.join_networks(
                accumulated_network, result_state, i, target)

            # don't include output gates as possible inputs
            # new Ni = Ng - No + Ni
            No = accumulated_network.No
            inputs = accumulated_network.activation_matrix[:-No, :]
            inputs = BitPackedMatrix(inputs, Ne=self.Ne, Ni=inputs.shape[0])

            self.learned_targets.append(target)

        # reorder the outputs to match the supplied target order
        # NOTE: This is why output gates are not included as possible inputs
        self.reorder_network_outputs(accumulated_network.representation)

        return LearnerResult(
            network=accumulated_network.representation,
            partial_networks=[r.representation for r in opt_results],
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

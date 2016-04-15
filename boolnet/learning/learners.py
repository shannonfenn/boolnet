from copy import copy
from collections import namedtuple
import logging
import numpy as np
import boolnet.bintools.functions as gf
from boolnet.bintools.packing import (
    unpack_bool_matrix, unpack_bool_vector, BitPackedMatrix)
from boolnet.learning.networkstate import StandardBNState
import boolnet.learning.feature_selection as mfs


LearnerResult = namedtuple('LearnerResult', [
    'network', 'partial_networks', 'best_errors', 'best_iterations',
    'final_iterations', 'target_order', 'feature_sets', 'restarts'])


def per_tgt_err_stop_criterion(bit):
    return lambda ev, _: ev.function_value(gf.PER_OUTPUT)[bit] <= 0


def guiding_func_stop_criterion():
    return lambda _, error: error <= 0


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
        self.No = self.problem_matrix.shape[0] - self.Ni
        self.input_matrix, self.target_matrix = np.split(
            self.problem_matrix, [self.Ni])
        self.Ne = self.problem_matrix.Ne
        # Optimiser
        self.optimiser = optimiser
        self.opt_params = copy(parameters['optimiser'])
        gf_name = self.opt_params['guiding_function']
        self.guiding_func_id = gf.function_from_name(gf_name)
        self.opt_params['guiding_function'] = (
            lambda x: x.function_value(self.guiding_func_id))

        if self.guiding_func_id not in gf.scalar_functions():
            raise ValueError('Invalid guiding function: {}'.format(gf_name))
        if max(self.node_funcs) > 15 or min(self.node_funcs) < 0:
            raise ValueError('\'node_funcs\' must come from [0, 15]: {}'.
                             format(self.node_funcs))

    def run(self, optimiser, parameters):
        self._setup(optimiser, parameters)

        gates = self.gate_generator(self.budget, self.Ni, self.node_funcs)
        state = StandardBNState(gates, self.problem_matrix)

        self.opt_params['stopping_criterion'] = guiding_func_stop_criterion()

        opt_result = self.optimiser.run(state, self.opt_params)

        return LearnerResult(
            network=opt_result.representation,
            best_errors=[opt_result.error],
            best_iterations=[opt_result.best_iteration],
            final_iterations=[opt_result.iteration],
            restarts=[opt_result.restarts],
            target_order=None,
            feature_sets=None,
            partial_networks=[])


class StratifiedLearner(BasicLearner):

    def _setup(self, optimiser, parameters):
        super()._setup(optimiser, parameters)
        # Required
        self.mfs_fname = parameters['inter_file_base']
        # Optional
        self.auto_target = parameters.get('auto_target', False)
        self.use_minfs_selection = parameters.get('minfs_masking', False)
        self.keep_files = parameters.get('keep_files', False)
        self.fabcpp_opts = parameters.get('fabcpp_options', {})
        # Initialise
        self.No, _ = self.target_matrix.shape
        self.remaining_budget = self.budget
        self.learned_targets = []
        self.feature_sets = np.empty((self.No, self.No),
                                     dtype=list)

    def _determine_next_target(self, strata, inputs):
        all_targets = set(range(self.No))
        not_learned = list(all_targets.difference(self.learned_targets))
        if self.auto_target:
            # find minFS for all unlearned targets
            self._record_feature_sets(strata, inputs, not_learned)
            feature_sets_sizes = [len(l) for l in
                                  self.feature_sets[strata][not_learned]]
            indirect_simplest_target = np.argmin(feature_sets_sizes)
            return not_learned[indirect_simplest_target]
        elif self.use_minfs_selection:
            target = min(not_learned)
            # find minFS for just the next target
            self._record_feature_sets(strata, inputs, [target])
            return target
        else:
            return min(not_learned)

    def _record_feature_sets(self, strata, inputs, targets):
        # keep a log of the feature sets found at each iteration
        for t in targets:
            fs = self._get_fs(inputs, t, strata)
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

    def make_partial_problem(self, strata, target_index, inputs):
        # determine next budget
        target = self.target_matrix[target_index]

        if self.use_minfs_selection:
            # subsample the inputs for below
            fs = self.feature_sets[strata, target_index]
            inputs = inputs[fs]

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
            new_input_map = self.feature_sets[strata][target_index].tolist()
            # difference in input sizes plus # of non-output base gates
            # offset = base.Ni - new.Ni + base.Ng - base.No
            offset = base.Ni + base.Ng - base.No
            new_gate_map = list(range(offset,
                                      offset + new.Ng - new.No))
            new_output_map = list(range(offset + new.Ng - new.No + base.No,
                                        offset + new.Ng + base.No))
            sources_map = new_input_map + new_gate_map + new_output_map

            # apply to all but the last column (it is for the transfer
            # functions) of the gate matrix
            # remapped_new_gates = new.gates.copy()
            # numpy array since cython memoryview slicing is broken
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
            target = self._determine_next_target(i, inputs)

            partial_problem = self.make_partial_problem(i, target, inputs)

            gates = self.next_gates(i, partial_problem)
            state = StandardBNState(gates, partial_problem)

            # generate an end condition based on the error
            criterion = guiding_func_stop_criterion()
            self.opt_params['stopping_criterion'] = criterion

            # optimise
            partial_result = self.optimiser.run(state, self.opt_params)
            # record result
            opt_results.append(partial_result)

            # build up final network by inserting this partial result
            result_state = StandardBNState(
                partial_result.representation.gates, partial_problem)
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

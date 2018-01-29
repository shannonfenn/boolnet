from copy import copy
from collections import namedtuple
import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

import boolnet.network.algorithms as alg
from boolnet.utils import PackedMatrix
from boolnet.network.networkstate import BNState
from time import time


def prefilter_features(Ni, strata_sizes, strata, targets, fs_table, method):
    if strata == 0:
        prev_strata_range = []
    else:
        strata_limits = (sum(strata_sizes[:strata-1]) + Ni,
                         sum(strata_sizes[:strata]) + Ni)
        prev_strata_range = list(range(*strata_limits))

    input_range = list(range(Ni))

    if method == 'all':
        # bound on Nf: none
        return list(range(sum(strata_sizes[:strata]) + Ni))
    elif method == 'prev-strata':
        # bound on Nf: L
        # Note: no guarantee that the prior strata contains a valid fs
        raise NotImplementedError(
            'Not implemented since can result in invalid minFS instance.')
    elif method == 'prev-strata+input':
        # bound on Nf: L + Ni
        return input_range + prev_strata_range
    elif method == 'prev-strata+prev-fs':
        # bound on Nf: L + Ni
        # this actually produces a list of feature matrices
        if strata == 0:
            # there is no prior fs
            return input_range
        else:
            filtered_features = []
            for t in targets:
                prev_fs = fs_table[strata - 1][t]
                if prev_fs is None:
                    prev_fs = set()
                else:
                    prev_fs = set(prev_fs)
                f = prev_fs.union(prev_strata_range)
                f = sorted(f)
                filtered_features.append(f)
            return filtered_features
    elif method == 'prev-strata+prev-fs+input':
        # bound on Nf: L + 2xNi
        # this actually produces a list of feature matrices
        if strata == 0:
            # there is no prior fs
            return input_range
        else:
            filtered_features = []
            for t in targets:
                prev_fs = fs_table[strata - 1][t]
                if prev_fs is None:
                    prev_fs = set()
                else:
                    prev_fs = set(prev_fs)
                f = prev_fs.union(input_range, prev_strata_range)
                f = sorted(f)
                filtered_features.append(f)
            return filtered_features
    else:
        raise ValueError('Invalid prefilter: {}'.format(method))


def ranked_fs_helper(Xp, Yp, Ne, Ni, strata_sizes, strata, targets, fs_table,
                     prefilter_method, metric, solver, solver_params,
                     tie_handling='random', provide_prior_soln=False):

    F_in = prefilter_features(Ni, strata_sizes, strata, targets, fs_table,
                              prefilter_method)

    # unpack inputs to minFS solver
    if isinstance(F_in[0], int):
        mfs_X = pk.unpackmat(Xp[F_in, :], Ne)
    else:
        mfs_X = [pk.unpackmat(Xp[fs, :], Ne) for fs in F_in]
    mfs_Y = pk.unpackmat(Yp[targets, :], Ne)

    ranking, result_feature_sets = mfs.ranked_feature_sets(
        mfs_X, mfs_Y, metric, solver, solver_params)

    # remap feature sets using given feature indices
    if isinstance(F_in[0], int):
        fs_maps = (F_in for i in range(len(targets)))
    else:
        fs_maps = F_in
    for fs, fsmap in zip(result_feature_sets, fs_maps):
        for i, f in enumerate(fs):
            fs[i] = fsmap[f]

    # update feature set table
    for t, fs in zip(targets, result_feature_sets):
        if len(fs) == 0:
            # replace empty feature sets
            fs_table[strata, t] = list(range(Xp.shape[0]))
        else:
            fs_table[strata, t] = fs

    if tie_handling == 'random':
        # randomly pick from top ranked targets
        next_targets = targets[np.random.choice(np.where(ranking == 0)[0])]
    elif tie_handling == 'min_depth':
        # can do using a tuple to sort where the second element is the
        # inverse of the largest feature (i.e. the greatest depth)
        raise NotImplementedError('min_depth tie handling not available.')
    elif tie_handling == 'all':
        # return all top ranked targets
        next_targets = targets[np.where(ranking == 0)[0]]
    else:
        raise ValueError('Invalid choice for tie_handling.')

    return next_targets.tolist()


class Learner():

    def _setup(self, parameters):
        # Gate generation
        self.model_generator = parameters['model_generator']
        self.budget = parameters['network']['Ng']
        # get target order
        self.target_order = parameters['target_order']
        # Instance
        self.D = parameters['training_set']
        self.X, self.Y = np.split(self.D, [self.D.Ni])
        # Optional
        self.minfs_solver = parameters.get('minfs_solver', 'cplex')
        self.minfs_params = parameters.get('minfs_solver_params', {})
        self.minfs_tie_handling = parameters.get('minfs_tie_handling',
                                                 'random')
        self.use_minfs_selection = parameters.get('minfs_masking', False)
        self.minfs_metric = parameters.get('minfs_selection_metric', None)
        self.minfs_prefilter = parameters.get('minfs_prefilter', None)
        self.shrink_subnets = parameters.get('shrink_subnets', True)
        self.reuse_gates = parameters.get('reuse_gates', False)
        # Initialise
        self.remaining_budget = self.budget
        self.strata_sizes = []
        self.learned_targets = []
        self.feature_sets = np.empty((self.D.No, self.D.No), dtype=list)

    def determine_next_target(self, strata, inputs):
        if self.target_order is None:
            # get unlearned targets
            to_learn = np.setdiff1d(range(self.D.No), self.learned_targets)
        elif self.use_minfs_selection:
            # get next target from given ordering
            to_learn = [self.target_order[strata]]
        else:
            return self.target_order[strata]
        next_target = ranked_fs_helper(
            inputs, self.Y, self.D.Ne, self.D.Ni,
            self.strata_sizes, strata, to_learn, self.feature_sets,
            self.minfs_prefilter, self.minfs_metric, self.minfs_solver,
            self.minfs_params, self.minfs_tie_handling, False)

        return next_target

    def make_partial_instance(self, strata, target_index, inputs):
        if self.use_minfs_selection:
            # subsample the inputs for below
            fs = self.feature_sets[strata, target_index]
            inputs = inputs[fs]

        target = self.Y[target_index]

        return PackedMatrix(np.vstack((inputs, target)),
                            Ne=self.D.Ne, Ni=inputs.shape[0])

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

        D_new = PackedMatrix(np.vstack((self.X, new_target)),
                             Ne=self.D.Ne, Ni=self.D.Ni)

        return BNState(accumulated_gates, D_new)

    def reorder_network_outputs(self, network):
        # all non-output gates are left alone, and the output gates are
        # reordered by the inverse permutation of "learned_targets"
        new_out_order = mfs.inverse_permutation(self.learned_targets)
        new_gates = np.array(network.gates)
        No = network.No
        new_gates[-No:, :] = new_gates[-No:, :][new_out_order]
        network.set_gates(new_gates)

    def run(self, optimiser, parameters, verbose=False):
        # setup accumulated network
        # loop:
        #   make partial network
        #   optimise it
        #   hook into accumulated network
        # reorganise outputs

        self._setup(parameters)

        opt_results = []

        inputs = self.X.copy()

        # make a state with Ng = No = 0 and set the inp mat = self.input_matrix
        accumulated_network = BNState(np.empty((0, 3)), inputs)

        optimisation_times = []
        other_times = []

        for i in range(self.D.No):
            if verbose:
                print('Strata {}'.format(i))
            t0 = time()

            if verbose:
                print('  Determining target...')

            # determine next target index
            target = self.determine_next_target(i, inputs)

            if verbose:
                print('  done. Target {} selected.'.format(target))

            D_partial = self.make_partial_instance(i, target, inputs)

            # ### build state to be optimised ### #
            # next batch of gates
            size = self.remaining_budget // (self.D.No - i)
            gates = self.model_generator(size, D_partial.Ni,
                                         D_partial.No)
            state = BNState(gates, D_partial)

            t1 = time()

            if verbose:
                print('  time taken: {}'.format(t1 - t0))
                print('  Optimising...')

            # optimise
            partial_result = optimiser.run(state, parameters['optimiser'])

            t2 = time()

            if verbose:
                print('  done. Error: {}'.format(partial_result.error))
                print('  time taken: {}'.format(t2 - t1))

            # record result
            opt_results.append(partial_result)

            net = partial_result.representation
            if self.shrink_subnets:
                # percolate
                new_gates = alg.filter_connected(net.gates, net.Ni, net.No)
                result_state = BNState(new_gates, D_partial)
                if self.reuse_gates:
                    size = new_gates.shape[0]
            else:
                result_state = BNState(net.gates, D_partial)

            # update node budget and strata sizes
            self.remaining_budget -= size
            self.strata_sizes.append(result_state.Ng - result_state.No)

            accumulated_network = self.join_networks(
                accumulated_network, result_state, i, target)

            # don't include output gates as possible inputs
            # new Ni = Ng - No + Ni
            No = accumulated_network.No
            inputs = accumulated_network.activation_matrix[:-No, :]
            inputs = PackedMatrix(inputs, Ne=self.D.Ne, Ni=inputs.shape[0])

            self.learned_targets.append(target)

            t3 = time()
            optimisation_times.append(t2 - t1)
            other_times.append(t3 - t2 + t1 - t0)

        # reorder the outputs to match the original target order
        # NOTE: This is why output gates are not included as possible inputs
        self.reorder_network_outputs(accumulated_network.representation)

        return {
            'network': accumulated_network.representation,
            'target_order': self.learned_targets,
            'extra': {
                'partial_networks': [r.representation for r in opt_results],
                'best_err': [r.error for r in opt_results],
                'best_step': [r.best_iteration for r in opt_results],
                'steps': [r.iteration for r in opt_results],
                'feature_sets': self.feature_sets,
                'restarts': [r.restarts for r in opt_results],
                'opt_time': optimisation_times,
                'other_time': other_times,
                'strata_sizes': self.strata_sizes,
                }
            }

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



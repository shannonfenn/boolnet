import numpy as np

import minfs.feature_selection as mfs

import boolnet.network.algorithms as alg
from boolnet.utils import PackedMatrix
from boolnet.network.networkstate import BNState
from boolnet.learners.stratified import ranked_fs_helper
from time import time
import itertools


class Learner:

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
        self.minfs_params = parameters.get('minfs_params', {})
        self.use_minfs_selection = parameters.get('minfs_masking', False)
        self.minfs_prefilter = parameters.get('prefilter', None)
        self.shrink_subnets = parameters.get('shrink_subnets', True)
        # Initialise
        self.remaining_budget = self.budget
        self.strata_sizes = []
        self.learned_targets = []
        self.feature_sets = np.empty((self.D.No, self.D.No), dtype=list)

    def remaining_targets(self):
        flat_learned = list(
                itertools.chain.from_iterable(self.learned_targets))
        return np.setdiff1d(range(self.D.No), flat_learned)

    def determine_next_targets(self, strata, inputs):
        if self.auto_target:
            to_learn = self.remaining_targets()
        elif self.use_minfs_selection:
            # get next targets from given ordering
            to_learn = np.array([self.target_order[strata]])
        else:
            return self.target_order[strata]

        if strata == 0:
            prev_fsets = self.feature_sets[0]
        else:
            prev_fsets = self.feature_sets[strata - 1]

        ranking, next_fsets = ranked_fs_helper(
            inputs, self.Y, self.D.Ni, self.strata_sizes, to_learn,
            prev_fsets, self.minfs_prefilter, self.minfs_params)
        self.feature_sets[to_learn] = next_fsets
        next_targets = to_learn[np.where(ranking == 0)[0]]

        print('next: ', next_targets)

        return next_targets

    def _combined_feature_set(self, strata, target_indices):
        fs = set()
        for t in target_indices:
            fs.update(self.feature_sets[strata, t])
        return sorted(fs)

    def make_partial_instance(self, strata, target_indices, inputs):
        if self.use_minfs_selection:
            # subsample the inputs for below
            fs = self._combined_feature_set(strata, target_indices)
            inputs = inputs[fs]

        targets = self.Y[target_indices]

        return PackedMatrix(np.vstack((inputs, targets)),
                            Ne=self.D.Ne, Ni=inputs.shape[0])

    def join_networks(self, base, new, strata, target_indices):
        # simple: build up a map for all sources, for sources after the
        # original minfs input just have them as + Ni_offset (whatever that is,
        # might including old.Ng) and for the former ones use the fs mapping

        if self.use_minfs_selection:
            # build a map for replacing sources
            new_input_map = self._combined_feature_set(strata, target_indices)
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

        Y_new = np.vstack((base.target_matrix, new.target_matrix))

        D_new = PackedMatrix(np.vstack((self.X, Y_new)),
                             Ne=self.D.Ne, Ni=self.D.Ni)

        return BNState(accumulated_gates, D_new)

    def reorder_network_outputs(self, network):
        # all non-output gates are left alone, and the output gates are
        # reordered by the inverse permutation of "learned_targets"
        No = network.No
        flat_learned = list(
            itertools.chain.from_iterable(self.learned_targets))
        print(list(self.learned_targets))
        print(list(flat_learned))
        new_out_order = mfs.inverse_permutation(flat_learned)
        new_gates = np.array(network.gates)
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

        # make a state with Ng = No = 0 and set the inp mat = self.X
        accumulated_network = BNState(np.empty((0, 3)), inputs)

        optimisation_times = []
        other_times = []

        strata = 0
        while len(self.remaining_targets()) > 0:
            if verbose:
                print('Strata {}'.format(strata))
            t0 = time()

            if verbose:
                print('  Determining target...')

            # determine next target index
            targets = self.determine_next_targets(strata, inputs)

            if verbose:
                print('  done. Targets selected: {} .'.format(targets))

            D_partial = self.make_partial_instance(strata, targets, inputs)

            # ### build state to be optimised ### #
            # next batch of gates
            size = (self.remaining_budget * len(targets) //
                    len(self.remaining_targets()))
            gates = self.model_generator(size, D_partial.Ni, D_partial.No)
            state = BNState(gates, D_partial)

            t1 = time()

            if verbose:
                print('  time taken: {}'.format(t1 - t0))
                print('  Optimising...')

            # optimise
            partial_result = optimiser.run(state)

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
                if self.reuse_gates:
                    size = new_gates.shape[0]
            else:
                new_gates = net.gates
            result_state = BNState(new_gates, D_partial)

            # update node budget and strata sizes
            self.remaining_budget -= size
            self.strata_sizes.append(result_state.Ng - result_state.No)

            accumulated_network = self.join_networks(
                accumulated_network, result_state, strata, targets)

            # don't include output gates as possible inputs
            # new Ni = Ng - No + Ni
            No = accumulated_network.No
            inputs = accumulated_network.activation_matrix[:-No, :]
            inputs = PackedMatrix(inputs, Ne=self.D.Ne, Ni=inputs.shape[0])

            self.learned_targets.append(targets)

            t3 = time()
            optimisation_times.append(t2 - t1)
            other_times.append(t3 - t2 + t1 - t0)

            strata += 1

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

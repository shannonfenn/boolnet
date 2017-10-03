import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.utils import PackedMatrix
from boolnet.network.networkstate import BNState
from time import time


class Learner:
    def _setup(self, parameters):
        # Gate generation
        self.model_generator = parameters['model_generator']
        self.budget = parameters['network']['Ng']
        self.remaining_budget = self.budget
        # Instance
        self.D = parameters['training_set']
        self.X, self.Y = np.split(self.D, [self.D.Ni])
        # Optional feature selection params
        self.use_minfs_selection = parameters.get('minfs_masking', False)
        self.minfs_metric = parameters.get('minfs_selection_metric', None)
        self.minfs_params = parameters.get('minfs_solver_params', {})
        self.minfs_solver = parameters.get('minfs_solver', 'cplex')
        self.feature_sets = np.empty(self.D.No, dtype=list)

    def next_subnet(self, t):
        size = self.remaining_budget // (self.D.No - t)
        self.remaining_budget -= size
        fs = self.feature_sets[t]
        return self.model_generator(size, len(fs), 1)

    def join_networks(self, base, new, t):
        if self.use_minfs_selection:
            # build a map for replacing sources
            new_input_map = list(self.feature_sets[t])
        else:
            # build a map for replacing sources
            new_input_map = list(range(self.D.Ni))
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
                # gate.size-1 since the last entry is the transfer function
                gate[i] = sources_map[gate[i]]

        accumulated_gates = np.vstack((base.gates[:-base.No, :],
                                       remapped_new_gates[:-new.No, :],
                                       base.gates[-base.No:, :],
                                       remapped_new_gates[-new.No:, :]))

        new_target = np.vstack((base.target_matrix, new.target_matrix))

        new_D = PackedMatrix(np.vstack((self.X, new_target)),
                             Ne=self.D.Ne, Ni=self.D.Ni)
        return BNState(accumulated_gates, new_D)

    def get_feature_sets(self):
        # default - in case of empty fs or fs-selection not enabled
        for t in range(self.D.No):
            self.feature_sets[t] = list(range(self.D.Ni))

        if self.use_minfs_selection:
            # unpack inputs to minFS solver
            mfs_features = pk.unpackmat(self.X, self.D.Ne)
            mfs_targets = pk.unpackmat(self.Y, self.D.Ne)
            _, F = mfs.ranked_feature_sets(
                mfs_features, mfs_targets, self.minfs_metric,
                self.minfs_solver, self.minfs_params)
            for t, fs in enumerate(F):
                if fs:
                    self.feature_sets[t] = fs

    def make_partial_instance(self, target_index):
        target = self.Y[target_index]
        fs = self.feature_sets[target_index]
        Xsub = self.X[fs, :]
        return PackedMatrix(np.vstack((Xsub, target)),
                            Ne=self.D.Ne, Ni=len(fs))

    def run(self, optimiser, parameters, verbose=False):
        t0 = time()

        self._setup(parameters)

        opt_results = []
        optimisation_times = []
        other_times = []

        # make a state with Ng = No = 0 and set the inp mat = self.input_matrix
        accumulated_network = BNState(np.empty((0, 3)), self.X)

        if verbose:
            print('Getting feature sets...')

        self.get_feature_sets()

        feature_selection_time = time() - t0

        if verbose:
            print('done. Time taken: {}'.format(feature_selection_time))

        for target_index in range(self.D.No):
            t0 = time()

            D_partial = self.make_partial_instance(target_index)

            # build the network state
            gates = self.next_subnet(target_index)
            state = BNState(gates, D_partial)

            t1 = time()

            if verbose:
                print('Target {} Optimising...'.format(target_index))
            # run the optimiser
            partial_result = optimiser.run(state, parameters['optimiser'])
            t2 = time()

            if verbose:
                print('done. Time taken: {}'.format(t2 - t1))

            # record result
            opt_results.append(partial_result)

            # build up final network by inserting this partial result
            result_state = BNState(partial_result.representation.gates,
                                   D_partial)
            accumulated_network = self.join_networks(
                accumulated_network, result_state, target_index)

            t3 = time()
            optimisation_times.append(t2 - t1)
            other_times.append(t3 - t2 + t1 - t0)

        return {
            'network': accumulated_network.representation,
            'target_order': list(range(self.D.No)),
            'extra': {
                'partial_networks': [r.representation for r in opt_results],
                'best_err': [r.error for r in opt_results],
                'best_step': [r.best_iteration for r in opt_results],
                'steps': [r.iteration for r in opt_results],
                'feature_sets': self.feature_sets,
                'restarts': [r.restarts for r in opt_results],
                'opt_time': optimisation_times,
                'other_time': other_times,
                'fs_time': feature_selection_time
                }
            }

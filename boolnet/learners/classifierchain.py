import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.utils import PackedMatrix, order_from_rank
from boolnet.network.networkstate import BNState
from time import time


class Learner:

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

    def make_partial_instance(self, X, Y, target_index, chain):
        target = Y[target_index]
        Xsub = Y[chain, :]
        return PackedMatrix(np.vstack((X, Xsub, target)),
                            Ne=X.Ne, Ni=X.shape[0] + Xsub.shape[0])

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

    def get_target_order(self, X, Y, Ne, mfs_metric, mfs_solver, mfs_params,
                         mfs_tie_handling, verbose):
        if verbose:
            print('Determining target order...')
        # determine the target order by ranking feature sets
        mfs_X = pk.unpackmat(X, Ne)
        mfs_Y = pk.unpackmat(Y, Ne)

        # use external solver for minFS
        rank, feature_sets = mfs.ranked_feature_sets(
            mfs_X, mfs_Y, mfs_metric, mfs_solver, mfs_params)

        if mfs_tie_handling == 'random':
            # randomly pick from possible exact orders
            return order_from_rank(rank)
        elif mfs_tie_handling == 'min_depth':
            # can do using a tuple to sort where the second element is the
            # inverse of the largest feature (i.e. the greatest depth)
            raise NotImplementedError(
                'min_depth tie breaking not implemented.')
        else:
            raise ValueError('Invalid choice for tie_handling.')

        if verbose:
            print('done. Time taken: {}'.format(time() - t0))

    def run(self, optimiser, parameters, verbose=False):
        t0 = time()

        # Instance
        D = parameters['training_set']
        X, Y = np.split(D, [D.Ni])
        Ni, No, Ne = D.Ni, D.No, D.Ne

        # Gate generation
        model_generator = parameters['model_generator']
        total_budget = parameters['network']['Ng']
        budgets = spacings(total_budget, No)

        # get target order
        target_order = parameters['target_order']
        if target_order is None:
            # these parameters only required if auto-targetting
            mfs_solver = parameters.get('minfs_solver', 'cplex')
            mfs_metric = parameters['minfs_selection_metric']
            mfs_params = parameters.get('minfs_solver_params', {})
            mfs_tie_handling = parameters.get('minfs_tie_handling', 'random')
            target_order = self.get_target_order(
                    X, Y, Ne, mfs_metric, mfs_solver, mfs_params,
                    mfs_tie_handling, verbose)

        # build the network state
        gates = model_generator(budget, Ni, No)

        # reorder problem matrix
        outputs = D[-No:, :]
        outputs[:] = outputs[target_order, :]

        state = BNState(gates, D)

        t1 = time()

        if verbose:
            print('Optimising...')

        # run the optimiser
        opt_result = optimiser.run(state, parameters['optimiser'])
        t2 = time()

        if verbose:
            print('done. Time taken: {}'.format(t2 - t1))

        # undo ordering
        inverse_order = inverse_permutation(target_order)
        outputs[:] = outputs[inverse_order, :]

        gates = np.array(opt_result.representation.gates)
        out_gates = gates[-No:, :]
        out_gates[:] = out_gates[inverse_order, :]
        opt_result.representation.set_gates(gates)

        # TO REORDER THE NETWORK OUTPUTS JUST MAKE AND EXTRA GROUP OF OR GATES
        # THAT EACH TAKE THE CORRECT OUTPUT NODE AS BOTH INPUTS - DON'T RESTRICT
        # THE BUDGET THOUGH TO ERR ON THE SIDE OF BEING FAIR

        return {
            'network': opt_result.representation,
            'target_order': target_order,
            'extra': {
                'best_err': [opt_result.error],
                'best_step': [opt_result.best_iteration],
                'steps': [opt_result.iteration],
                'restarts': [opt_result.restarts],
                'opt_time': t2-t1,
                'other_time': t1-t0,
                }
            }

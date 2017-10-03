import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.utils import order_from_rank, inverse_permutation
from boolnet.network.networkstate import BNState
from time import time


class Learner:
    def _setup(self, parameters):
        # Gate generation
        self.model_generator = parameters['model_generator']
        self.budget = parameters['network']['Ng']
        # get target order
        self.target_order = parameters['target_order']
        if self.target_order is None:
            # this key is only required if auto-targetting
            self.minfs_metric = parameters['minfs_selection_metric']
        # Optional minfs solver time limit
        self.minfs_params = parameters.get('minfs_solver_params', {})
        self.minfs_solver = parameters.get('minfs_solver', 'cplex')
        self.minfs_tie_handling = parameters.get('minfs_tie_handling',
                                                 'random')

    def run(self, optimiser, parameters, verbose=False):
        t0 = time()
        self._setup(parameters)

        # Instance
        D = parameters['training_set']
        X, Y = np.split(D, [D.Ni])
        Ni, No, Ne = D.Ni, D.No, D.Ne

        if self.target_order is None:
            if verbose:
                print('Determining target order...')
            # determine the target order by ranking feature sets
            mfs_features = pk.unpackmat(X, Ne)
            mfs_targets = pk.unpackmat(Y, Ne)

            # use external solver for minFS
            rank, feature_sets = mfs.ranked_feature_sets(
                mfs_features, mfs_targets, self.minfs_metric,
                self.minfs_solver, self.minfs_params)

            if self.minfs_tie_handling == 'random':
                # randomly pick from possible exact orders
                self.target_order = order_from_rank(rank)
            elif self.minfs_tie_handling == 'min_depth':
                # can do using a tuple to sort where the second element is the
                # inverse of the largest feature (i.e. the greatest depth)
                raise NotImplementedError(
                    'min_depth tie breaking not implemented.')
            else:
                raise ValueError('Invalid choice for tie_handling.')

            if verbose:
                print('done. Time taken: {}'.format(time() - t0))

        # build the network state
        gates = self.model_generator(self.budget, Ni, No)

        # reorder problem matrix
        outputs = D[-No:, :]
        outputs[:] = outputs[self.target_order, :]

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
        inverse_order = inverse_permutation(self.target_order)
        outputs[:] = outputs[inverse_order, :]

        gates = np.array(opt_result.representation.gates)
        out_gates = gates[-No:, :]
        out_gates[:] = out_gates[inverse_order, :]
        opt_result.representation.set_gates(gates)

        return {
            'network': opt_result.representation,
            'target_order': self.target_order,
            'extra': {
                'best_err': [opt_result.error],
                'best_step': [opt_result.best_iteration],
                'steps': [opt_result.iteration],
                'restarts': [opt_result.restarts],
                'opt_time': t2-t1,
                'other_time': t1-t0,
                }
            }

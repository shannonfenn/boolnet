import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.network.networkstate import BNState
from time import time


def run(self, optimiser, parameters):
    t0 = time()

    # Gate generation
    model_generator = parameters['model_generator']
    budget = parameters['network']['Ng']
    # get target order
    target_order = parameters['target_order']
    if target_order is None:
        # this key is only required if auto-targetting
        mfs_metric = parameters['minfs_selection_metric']
    # Optional minfs solver time limit
    mfs_params = parameters.get('minfs_solver_params', {})
    mfs_solver = parameters.get('minfs_solver', 'cplex')
    mfs_tie_handling = parameters.get('minfs_tie_handling', 'random')

    # Instance
    D = parameters['training_set']
    X, Y = np.split(D, [D.Ni])
    Ni, No, Ne = D.Ni, D.No, D.Ne

    if target_order is None:
        # determine the target order by ranking feature sets
        mfs_X = pk.unpackmat(X, Ne)
        mfs_Y = pk.unpackmat(Y, Ne)

        # use external solver for minFS
        rank, feature_sets, _ = mfs.ranked_feature_sets(
            mfs_X, mfs_Y, mfs_metric, mfs_solver, mfs_params)

        if mfs_tie_handling == 'random':
            # randomly pick from possible exact orders
            target_order = mfs.order_from_rank(rank)
        elif mfs_tie_handling == 'min_depth':
            # can do using a tuple to sort where the second element is the
            # inverse of the largest feature (i.e. the greatest depth)
            raise NotImplementedError(
                'min_depth tie breaking not implemented.')
        else:
            raise ValueError('Invalid choice for tie_handling.')

    # build the network state
    gates = model_generator(budget, Ni, No)

    # reorder problem matrix
    outputs = D[-No:, :]
    outputs[:] = outputs[target_order, :]

    state = BNState(gates, D)

    # run the optimiser
    t1 = time()
    opt_result = optimiser.run(state)
    t2 = time()

    # undo ordering
    inverse_order = mfs.inverse_permutation(target_order)
    outputs[:] = outputs[inverse_order, :]

    gates = np.array(opt_result.representation.gates)
    out_gates = gates[-No:, :]
    out_gates[:] = out_gates[inverse_order, :]
    opt_result.representation.set_gates(gates)

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

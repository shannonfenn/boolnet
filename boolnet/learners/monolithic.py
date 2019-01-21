import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.network.networkstate import BNState
from time import time


def run(optimiser, model_generator, network_params, training_set,
        target_order=None, minfs_params={}, tie_handling='random'):

    t0 = time()

    # Gate generation
    budget = network_params['Ng']

    # Instance
    D = training_set
    X, Y = np.split(D, [D.Ni])
    Ni, No, Ne = D.Ni, D.No, D.Ne

    if target_order is None:
        # determine the target order by ranking feature sets
        mfs_X = pk.unpackmat(X, Ne)
        mfs_Y = pk.unpackmat(Y, Ne)

        # use external solver for minFS
        rank, feature_sets, _ = mfs.ranked_feature_sets(
            mfs_X, mfs_Y, **minfs_params)

        if tie_handling == 'random':
            # randomly pick from possible exact orders
            target_order = mfs.order_from_rank(rank)
        elif tie_handling == 'min_depth':
            # can do using a tuple to sort where the second element is the
            # inverse of the largest feature (i.e. the greatest depth)
            raise NotImplementedError(
                'min_depth tie breaking not implemented.')
        else:
            raise ValueError('Invalid choice for tie_handling.')
    else:
        feature_sets = None

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
            'feature_sets': feature_sets,
            'best_err': [opt_result.error],
            'best_step': [opt_result.best_iteration],
            'steps': [opt_result.iteration],
            'restarts': [opt_result.restarts],
            'opt_time': t2-t1,
            'other_time': t1-t0,
            }
        }

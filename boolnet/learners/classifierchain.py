import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.utils import PackedMatrix, spacings
from boolnet.network.networkstate import BNState
from time import time


def join_networks(networks, order):
    # The first network doesn't need modification
    Ni = networks[0].Ni
    prev_outputs = [Ni + networks[0].Ng - 1]
    remapped_gate_batches = [np.array(networks[0].gates, copy=True)]
    for n in range(1, len(networks)):
        net = networks[n]
        sources_map = (list(range(Ni)) +
                       prev_outputs +
                       list(range(prev_outputs[-1] + 1,
                                  prev_outputs[-1] + 1 + net.Ng)))
        # apply to all but the last column (transfer functions) of the gate
        # matrix. Use numpy array: cython memoryview slicing is broken
        remapped_gates = np.array(net.gates, copy=True)
        for gate in remapped_gates:
            for i in range(gate.size - 1):
                # gate.size-1 since the last entry is the transfer function
                gate[i] = sources_map[gate[i]]
        remapped_gate_batches.append(remapped_gates)
        prev_outputs.append(sources_map[-1])

    # final set of OR gates for reordering output
    outputs = np.zeros((len(order), remapped_gate_batches[0].shape[1]),
                       dtype=remapped_gate_batches[0].dtype)
    outputs[:, -1] = 14  # OR function
    for i, o in enumerate(order):
        outputs[i, :-1] = prev_outputs[o]
    remapped_gate_batches.append(outputs)

    return np.vstack(remapped_gate_batches)


def make_partial_instance(X, Y, target_index, chain):
    target = Y[target_index]
    Xsub = Y[chain, :]
    return PackedMatrix(np.vstack((X, Xsub, target)),
                        Ne=X.Ne, Ni=X.shape[0] + Xsub.shape[0])


def minfs_target_order(X, Y, solver, metric, params, tie_handling):
    # determine the target order by ranking feature sets
    mfs_X = pk.unpackmat(X, X.Ne)
    mfs_Y = pk.unpackmat(Y, Y.Ne)

    # use external solver for minFS
    rank, feature_sets, _ = mfs.ranked_feature_sets(
        mfs_X, mfs_Y, metric, solver, params)

    if tie_handling == 'random':
        # randomly pick from possible exact orders
        return mfs.order_from_rank(rank), feature_sets
    else:
        raise ValueError('Invalid choice for tie_handling.')


def run(self, optimiser, parameters, verbose=False):
    t0 = time()

    if verbose:
        print('Initialising...')

    # Instance
    D = parameters['training_set']
    X, Y = np.split(D, [D.Ni])

    # Gate generation
    model_generator = parameters['model_generator']
    total_budget = parameters['network']['Ng']
    total_budget -= D.No  # We need OR gates at the end for tgt reordering
    budgets = spacings(total_budget, D.No)

    # get target order
    target_order = parameters['target_order']
    if target_order is None:
        # these parameters only required if auto-targetting
        mfs_solver = parameters.get('minfs_solver', 'cplex')
        mfs_metric = parameters['minfs_selection_metric']
        mfs_params = parameters.get('minfs_solver_params', {})
        mfs_tie_handling = parameters.get('minfs_tie_handling', 'random')
        target_order, feature_sets = minfs_target_order(
                X, Y, mfs_solver, mfs_metric, mfs_params, mfs_tie_handling)
    else:
        feature_sets = None

    opt_results = []
    optimisation_times = []
    other_times = []

    init_time = time() - t0

    if verbose:
        print('done. Time taken: {}'.format(init_time))
        print('Beginning training loop...')

    for i, budget in enumerate(budgets):
        t0 = time()

        target_index = target_order[i]

        D_i = make_partial_instance(X, Y, target_index, target_order[:i])

        # build the network state
        gates = model_generator(budget, D_i.Ni, D_i.No)

        state = BNState(gates, D_i)

        t1 = time()

        if verbose:
            print('Optimising target {}...'.format(target_index))
        # run the optimiser
        partial_result = optimiser.run(state)
        opt_results.append(partial_result)
        t2 = time()

        if verbose:
            print('done. Time taken: {}'.format(t2 - t1))

        other_times.append(t1 - t0)
        optimisation_times.append(t2 - t1)

    partial_networks = [r.representation for r in opt_results]
    accumulated_gates = join_networks(
        partial_networks, mfs.inverse_permutation(target_order))

    return {
        'network': BNState(accumulated_gates, D),
        'target_order': target_order,
        'extra': {
            'partial_networks': partial_networks,
            'best_err': [r.error for r in opt_results],
            'best_step': [r.best_iteration for r in opt_results],
            'steps': [r.iteration for r in opt_results],
            'feature_sets': feature_sets,
            'restarts': [r.restarts for r in opt_results],
            'opt_time': optimisation_times,
            'other_time': other_times,
            'init_time': init_time
            }
        }

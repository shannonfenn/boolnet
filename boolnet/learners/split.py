import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

from boolnet.utils import PackedMatrix, spacings
from boolnet.network.networkstate import BNState
from time import time


def join_networks(base, new, t, D, X, feature_sets, use_minfs_selection):
    if use_minfs_selection and feature_sets[t]:
        # build a map for replacing sources
        new_input_map = list(feature_sets[t])
    else:
        # build a map for replacing sources
        new_input_map = list(range(D.Ni))
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

    new_D = PackedMatrix(np.vstack((X, new_target)),
                         Ne=D.Ne, Ni=D.Ni)
    return BNState(accumulated_gates, new_D)


def get_feature_sets(D, X, Y, feature_sets, apply_mask, minfs_params):
    # default - in case of empty fs or fs-selection not enabled
    # for t in range(D.No):
    #     feature_sets[t] = list(range(D.Ni))

    if apply_mask:
        # unpack inputs to minFS solver
        mfs_features = pk.unpackmat(X, D.Ne)
        mfs_targets = pk.unpackmat(Y, D.Ne)
        _, F, _ = mfs.ranked_feature_sets(
            mfs_features, mfs_targets, **minfs_params)
        for t, fs in enumerate(F):
            feature_sets[t] = fs


def make_partial_instance(X, Y, feature_sets, target_index):
    target = Y[target_index]
    fs = feature_sets[target_index]
    if fs:
        Xsub = X[fs, :]
    else:
        Xsub = X
    return PackedMatrix(np.vstack((Xsub, target)),
                        Ne=X.Ne, Ni=Xsub.shape[0])


def run(optimiser, model_generator, network_params, training_set,
        target_order=None, minfs_params={}, apply_mask=False,
        early_terminate=True):
    t0 = time()

    D = training_set
    X, Y = np.split(D, [D.Ni])
    # Gate generation
    budgets = spacings(network_params['Ng'], D.No)
    # Optional feature selection params
    feature_sets = np.empty(D.No, dtype=list)

    opt_results = []
    optimisation_times = []
    other_times = []

    # make a state with Ng = No = 0 and set the inp mat = self.input_matrix
    accumulated_network = BNState(np.empty((0, 3)), X)

    get_feature_sets(D, X, Y, feature_sets, apply_mask, minfs_params)

    feature_selection_time = time() - t0

    if target_order is None:
        target_order = list(range(D.No))

    i = 0
    for target, size in zip(target_order, budgets):
        t1 = time()

        D_partial = make_partial_instance(X, Y, feature_sets, target)
        # build the network state
        gates = model_generator(size, D_partial.Ni, 1)
        state = BNState(gates, D_partial)

        # run the optimiser
        t2 = time()
        partial_result = optimiser.run(state)
        if early_terminate and partial_result.error > 0:
            raise ValueError(f'Step {i} Target {target} failed to memorise '
                             f'error: {partial_result.error} '
                             f'stage time: {time()-t1} total time: {time()-t0}')
        t3 = time()

        # record result
        opt_results.append(partial_result)

        # build up final network by inserting this partial result
        result_state = BNState(partial_result.representation.gates,
                               D_partial)
        accumulated_network = join_networks(
            accumulated_network, result_state, target, D, X,
            feature_sets, apply_mask)

        t4 = time()
        optimisation_times.append(t3 - t2)
        other_times.append(t4 - t3 + t2 - t1)
        i += 1

    return {
        'network': accumulated_network.representation,
        'target_order': target_order,
        'extra': {
            'partial_networks': [r.representation for r in opt_results],
            'best_err': [r.error for r in opt_results],
            'best_step': [r.best_iteration for r in opt_results],
            'steps': [r.iteration for r in opt_results],
            'feature_sets': feature_sets,
            'restarts': [r.restarts for r in opt_results],
            'opt_time': optimisation_times,
            'other_time': other_times,
            'fs_time': feature_selection_time
            }
        }

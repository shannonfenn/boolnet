import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

import boolnet.network.algorithms as alg
from boolnet.utils import PackedMatrix
from boolnet.network.networkstate import BNState
from time import time


def prefilter_features(Ni, strata_sizes, prev_fs, method):
    input_range = set(range(Ni))

    if strata_sizes:
        strata_limits = (sum(strata_sizes[:-1]) + Ni,
                         sum(strata_sizes) + Ni)
        prev_strata_range = list(range(*strata_limits))
    else:
        return sorted(input_range)

    if not method:
        # bound on Nf: none
        return list(range(sum(strata_sizes) + Ni))

    #   combination               bound on Nf
    # prev-strata+input             L + Ni
    # prev-strata+prev-fs           L + Ni
    # prev-strata+prev-fs+input     L + 2xNi

    filtered_features = set()

    if 'input' in method:
        filtered_features.update(input_range)

    if 'prev-strata' in method:
        # bound on Nf: L + Ni
        filtered_features.update(prev_strata_range)

    if 'prev-fs' in method and prev_fs:
        # bound on Nf: L + Ni
        # this actually produces a list of feature matrices
        filtered_features.update(prev_fs)

    return sorted(filtered_features)


def ranked_fs_helper(Xp, Yp, Ni, strata_sizes, targets, prev_fsets,
                     prefilter, minfs_params):
    if 'prev-fs' in prefilter:
        F_in = [prefilter_features(Ni, strata_sizes, prev_fsets[t], prefilter)
                for t in targets]
        mfs_X = [pk.unpackmat(Xp[fs, :], Xp.Ne) for fs in F_in]
        fs_maps = F_in
    else:
        F_in = prefilter_features(Ni, strata_sizes, None, prefilter)
        mfs_X = pk.unpackmat(Xp[F_in, :], Xp.Ne)
        fs_maps = (F_in for i in range(len(targets)))

    mfs_Y = pk.unpackmat(Yp[targets, :], Xp.Ne)

    ranking, feature_sets, _ = mfs.ranked_feature_sets(
        mfs_X, mfs_Y, **minfs_params)

    # remap feature sets using given feature indices
    for fs, fsmap in zip(feature_sets, fs_maps):
        for i, f in enumerate(fs):
            fs[i] = fsmap[f]

    return ranking, feature_sets


def single_fs_helper(Xp, yp, Ni, strata_sizes, prev_fs,
                     prefilter, solver, metric, solver_params):
    F_in = prefilter_features(Ni, strata_sizes, prev_fs, prefilter)
    X = pk.unpackmat(Xp[F_in, :], Xp.Ne)
    Y = pk.unpackmat(yp, Xp.Ne)

    fs, _, _ = mfs.best_feature_set(X, Y, metric, solver, solver_params)
    # remap feature sets using given feature indices
    fs = [F_in[f] for f in fs]
    return fs


def get_next_target(Xsub, Y, Ni, given_order, to_learn, strata_sizes,
                    prev_fsets, prefilter, use_filtering, tie_handling,
                    minfs_params):
    fsets = np.empty_like(prev_fsets)
    if given_order:
        # get the feature set now
        next_target = given_order.pop()
        if use_filtering:
            y = Y[next_target, :]
            fsets[next_target] = single_fs_helper(
                Xsub, y, Ni, strata_sizes, prev_fsets[next_target],
                prefilter, tie_handling, minfs_params)
    else:
        ranking, partial_fsets = ranked_fs_helper(
            Xsub, Y, Ni, strata_sizes, to_learn, prev_fsets,
            prefilter, tie_handling, minfs_params)
        # randomly pick from top ranked targets
        next_target = targets[np.random.choice(np.where(ranking == 0)[0])]
        fsets[to_learn] = partial_fsets
    return next_target, fsets


def partial_instance(Xsub, Y, t, feature_sets, apply_mask):
    if apply_mask:
        fs = feature_sets[t]
        if len(fs) > 0:     # empty fs should pass inputs through
            Xsub = Xsub[fs]
    return PackedMatrix(np.vstack((Xsub, Y[t, :])), Xsub.Ne, Xsub.shape[0])


def join_networks(base, new, fs, apply_mask, shrink):
    # simple: build up a map for all sources, for sources after the
    # original minfs input just have them as + Ni_offset (whatever that is,
    # might including old.Ng) and for the former ones use the fs mapping
    if apply_mask and fs:
        # build a map for replacing sources
        new_input_map = fs
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
    else:
        # Can just be tacked on since new.Ni = base.Ni + base.Ng - base.No
        remapped_new_gates = new.gates

    gates = np.vstack((base.gates[:-base.No, :],
                       remapped_new_gates[:-new.No, :],
                       base.gates[-base.No:, :],
                       remapped_new_gates[-new.No:, :]))

    if shrink:
        gates = alg.filter_connected(gates, base.Ni, base.No + new.No)

    D_new = PackedMatrix(np.vstack((base.input_matrix,
                                    base.target_matrix,
                                    new.target_matrix)),
                         Ne=base.Ne, Ni=base.Ni)

    return BNState(gates, D_new)


def reorder_network_outputs(network, learned_order):
    # all non-output gates are left alone, and the output gates are
    # reordered by the inverse permutation of "learned_targets"
    new_out_order = mfs.inverse_permutation(learned_order)
    new_gates = np.array(network.gates)
    No = network.No
    new_gates[-No:, :] = new_gates[-No:, :][new_out_order]
    network.set_gates(new_gates)


def run(optimiser, model_generator, network_params, training_set,
        target_order=None, tie_handling='random', minfs_params={},
        apply_mask=False, prefilter=None, shrink_subnets=False):
        # setup accumulated network
        # loop:
        #   make partial network
        #   optimise it
        #   hook into accumulated network
        # reorganise outputs

        # Unpack parameters
        X, Y = np.split(training_set, [training_set.Ni])

        # Initialise
        budget = network_params['Ng']
        strata_budgets = []
        strata_sizes = []
        learned_targets = []
        opt_results = []
        optimisation_times = []
        other_times = []
        fs_record = []
        to_learn = list(range(training_set.No))
        feature_sets = np.empty(training_set.No, dtype=list)
        # will be used as a stack
        target_order = reversed(target_order) if target_order else None
        # make an initial accumulator state with no gates
        accumulated = BNState(np.empty((0, 3)), X)

        while(to_learn):
            t0 = time()
            # don't include output gates as possible inputs
            # new Ni = Ng - No + Ni
            # NEED INPUTS TOO
            Xsub = accumulated.activation_matrix
            Xsub = Xsub[:accumulated.Ni+accumulated.Ng-accumulated.No, :]
            Xsub = PackedMatrix(Xsub, Ne=training_set.Ne, Ni=Xsub.shape[0])

            # determine next target index
            target, feature_sets = get_next_target(
                Xsub, Y, training_set.Ni, target_order, to_learn, strata_sizes,
                feature_sets, prefilter, filter, tie_handling, minfs_params)

            Dsub = partial_instance(Xsub, Y, target, feature_sets, apply_mask)

            # ### build state to be optimised ### #
            # next batch of gates
            size = (budget - accumulated.Ng) // len(to_learn)
            gates = model_generator(size, Dsub.Ni, Dsub.No)
            state = BNState(gates, Dsub)

            # optimise
            t1 = time()
            partial_result = optimiser.run(state)
            t2 = time()

            new_state = BNState(partial_result.representation.gates, Dsub)
            accumulated = join_networks(accumulated, new_state,
                                        feature_sets[target], apply_mask,
                                        shrink_subnets)

            opt_results.append(partial_result)
            learned_targets.append(target)
            fs_record.append(feature_sets)
            strata_budgets.append(size - Dsub.No)
            strata_sizes.append(
                accumulated.Ng - accumulated.No - sum(strata_sizes))

            optimisation_times.append(t2 - t1)
            other_times.append(time() - t2 + t1 - t0)

            to_learn = sorted(set(to_learn).difference(learned_targets))

        # reorder the outputs to match the original target order
        # NOTE: This is why output gates are not included as possible inputs
        reorder_network_outputs(accumulated.representation, learned_targets)

        # remove Nones
        fs_record = [[fs for fs in fsets if fs is not None]
                     for fsets in fs_record]

        return {
            'network': accumulated.representation,
            'target_order': learned_targets,
            'extra': {
                'partial_networks': [r.representation for r in opt_results],
                'best_err': [r.error for r in opt_results],
                'best_step': [r.best_iteration for r in opt_results],
                'steps': [r.iteration for r in opt_results],
                'feature_sets': fs_record,
                'restarts': [r.restarts for r in opt_results],
                'opt_time': optimisation_times,
                'other_time': other_times,
                'strata_budgets': strata_budgets,
                'strata_sizes': strata_sizes,
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

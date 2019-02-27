import numpy as np
import scipy.stats as stats
import bitpacking.packing as pk
import minfs.feature_selection as mfs
from pyitlib import discrete_random_variable as drv


def minfs_target_order(X, Y, minfs_params):
    # determine the target order by ranking feature sets
    mfs_X = pk.unpackmat(X, X.Ne)
    mfs_Y = pk.unpackmat(Y, Y.Ne)

    # use external solver for minFS
    rank, feature_sets, _ = mfs.ranked_feature_sets(
        mfs_X, mfs_Y, **minfs_params)

    return mfs.order_from_rank(rank), feature_sets


def conditional_entropy_matrix(Y):
    C = np.empty((Y.shape[1], Y.shape[1]))
    for i, yi in enumerate(Y.T):
        for j, yj in enumerate(Y.T):
            C[i, j] = drv.entropy_conditional(yi, yj)
    return C


def CEbCC(Y, variant=1):
    Y = pk.unpackmat(Y, Y.Ne)

    # Step 1. Initialize the remaining class labels set A
    A = list(range(Y.shape[1]))
    curriculum = []
    # Step 2. Estimate conditional entropy H X ( Y i | Y j ), i, j ∈ A
    H = conditional_entropy_matrix(Y)
    # Step 3. While | A | > 0
    while A:
        # Step 4.
        # for each element i in A , calculate SH
        if variant in [1, 2]:
            SH = H[A, :][:, A].sum(axis=1)
        elif variant in [3, 4]:
            SH = H[A, :][:, A].sum(axis=0)

        # Step 5. i ∗ = argmin SH ( i )      SKF note: break ties randomly
        if variant in [1, 3]:
            target = A[np.random.choice(np.flatnonzero(SH == SH.min()))]
        elif variant in [2, 4]:
            target = A[np.random.choice(np.flatnonzero(SH == SH.max()))]

        # Step 6. order ( | A | ) = i ∗
        if variant in [1, 4]:
            curriculum.insert(0, target)
        elif variant in [2, 3]:
            curriculum.append(target)
        # Step 7. A = A \ { i ∗ }
        A = [t for t in A if t != target]

    return curriculum, H


def effect_relation_matrix(Y):
    E = np.empty((Y.shape[1], Y.shape[1]))
    for i, yi in enumerate(Y.T):
        for j, yj in enumerate(Y.T):
            E[i, j] = np.abs(yj[yi == 0].mean() - yj[yi == 1].mean())
    np.fill_diagonal(E, 0)
    normaliser = E.sum(axis=0)
    normaliser[np.where(normaliser == 0)] = 1  # prevent division by zero
    return E / normaliser


def label_effects_rank(Y, t):
    E = effect_relation_matrix(Y)
    v = np.linalg.matrix_power(E, t).sum(axis=1)
    ranks = stats.rankdata(-v, 'min') - 1  # ranked high(0) to low(n-1)
    curricula = mfs.order_from_rank(ranks)
    return curricula, v

import numpy as np
import boolnet.utils as utils
import boolnet.learners.classifierchain as classifierchain
from time import time


def run(optimiser, model_generator, network_params, training_set,
        target_order=None, minfs_params={}, apply_mask=False):
    t0 = time()

    # sample the training set with replacement - Bagging
    indices = np.random.choice(training_set.Ne, size=training_set.Ne,
                               replace=True)
    training_set = utils.sample_packed(training_set, indices)

    # generate random ordering
    if target_order is None:
        target_order = np.random.permutation(training_set.No)

    t1 = time()
    # train CC and return
    results = classifierchain.run(
        optimiser, model_generator, network_params, training_set,
        target_order, minfs_params, apply_mask)

    results['extra']['init_time'] += t1 - t0

    return results

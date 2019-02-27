import numpy as np
import boolnet.learners.monolithic as monolithic
import boolnet.learners.stratified as stratified
import boolnet.learners.stratified_multipar as stratified_multipar
import boolnet.learners.split as split
import boolnet.learners.classifierchain as classifierchain
import boolnet.learners.classifierchain_plus as classifierchain_plus
import boolnet.learners.ecc_member as ecc_member
import boolnet.learners.curricula as cur
from time import time


SUBLEARNERS = {
    'monolithic': monolithic,
    'stratified': stratified,
    'stratmultipar': stratified_multipar,
    'split': split,
    'classifierchain': classifierchain,
    'classifierchain_plus': classifierchain_plus,
    'ecc_member': ecc_member,
    }


def run(training_set, sublearner, curricula_method, options={}, **kwargs):
    t0 = time()
    X, Y = np.split(training_set, [training_set.Ni])
    # get curricula
    if curricula_method == 'minfs':
        target_order, feature_sets = cur.minfs_target_order(X, Y, **options)
        extra = {'curricula_fs': feature_sets}
    elif curricula_method == 'CEbCC':
        target_order, H = cur.CEbCC(Y, **options)
        extra = {'curricula_H': H}
    elif curricula_method == 'label_effects':
        target_order, v, E = cur.label_effects_rank(Y, **options)
        extra = {'curricula_v': v,
                 'curricula_E': E}
    else:
        raise ValueError(f'Invalid curricula method: {curricula_method}')
    curricula_time = time() - t0

    extra['sublearner'] = sublearner
    extra['curricula_time'] = curricula_time

    results = SUBLEARNERS[sublearner].run(training_set=training_set, **kwargs)
    results['extra'].update(extra)

    return results

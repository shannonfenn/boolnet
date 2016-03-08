import numpy as np
from subprocess import check_call
import os
import re

# regex for matching feature lines
FEATURE_RE = re.compile('[0-9]+')

# from itertools import combinations
# def is_feature_set(fs, target):
#     # generator for the example index pairs to check
#     to_check = ((i1, i2) for i1, i2 in combinations(range(len(target)), 2)
#                 if target[i1] != target[i2])

#     # If all features are non-discriminating for any
#     # example pair then we do not have a feature set
#     return not any(np.array_equal(fs[e1], fs[e2]) for e1, e2 in to_check)


def abk_file(features, target, file_name):
    n_examples, n_features = features.shape

    feature_numbers = np.reshape(np.arange(n_features), (n_features, 1))

    abk_data = np.hstack((feature_numbers, features.T))

    sample_names = '\t'.join('s' + str(i) for i in range(n_examples))

    header = 'FEATURESINROWS\nTARGETPRESENT\nLAST\n{}\n{}\ndummy\t{}'.format(
        n_features, n_examples, sample_names)

    target_row = '\t' + '\t'.join(str(x) for x in target) + '\n'

    np.savetxt(file_name, abk_data, fmt='%d', delimiter='\t',
               header=header, footer=target_row, comments='')


def FABCPP_cmd_line(features, target, file_name_base, options, keep_files):
    abk_file_name = file_name_base + '.abk'
    log_file_name = file_name_base + '.log'
    err_file_name = file_name_base + '.err'
    out_file_name = file_name_base + '.sol'

    # check if files exist and raise exception

    # write abk file
    abk_file(features, target, abk_file_name)

    # ensure options is a mapping type if another 'False' type is given
    if not options:
        options = {}

    model = options.get('model', 6)
    alpha = options.get('a_min', 1)
    beta = options.get('b_min', 0)

    cmd_string = ['fabcpp', '-i', abk_file_name, '-o', file_name_base,
                  '-m', str(model), '-A', str(alpha), '-B', str(beta),
                  '-O', 'max:feature_degree']

    if model == 1:
        cover = options.get('cover', 'alfa')
        cmd_string += ['-y', cover]

    # run fabcpp
    with open(log_file_name, 'w') as log, open(err_file_name, 'w') as err:
        check_call(cmd_string, stdout=log, stderr=err)

    # parse output
    with open(out_file_name) as out:
        # pull lines from outfile which match the feature name regex
        k_feature_set = [int(l.strip()) for l in out
                         if FEATURE_RE.match(l.strip())]

    if not keep_files:
        os.remove(abk_file_name)
        os.remove(log_file_name)
        os.remove(err_file_name)
        os.remove(out_file_name)

    return np.array(k_feature_set)

    # TODO: Use model 5 to compute other FSs


def minimum_feature_set(features, target, file_name_base,
                        fabcpp_options, keep_files):
    ''' Takes a featureset matrix and target vector and finds a minimum FS.
    featureset  - assumed to be a 2D numpy array in example x feature format.
    target      - assumed to be a 1D numpy array of the same number of rows
                  as featureset
    returns     - a 1D numpy array of feature indices representing a FS
                  of minimal cardinality.'''

    return FABCPP_cmd_line(features, target, file_name_base,
                           fabcpp_options, keep_files)


def multi_target_minimum_feature_sets(features, targets, file_name_base,
                                      fabcpp_options, keep_files):
    ''' Takes a featureset matrix and target vector and finds a minimum FS.
    features    - 2D numpy array-like in example x feature format.
    target      - 2D numpy array-like of the same number of rows as features
    returns     - list of 1D numpy array-likes of feature indices representing
                  a feature set of minimal cardinality for each target.'''
    num_targets = targets.shape[1]
    feature_sets = [FABCPP_cmd_line(features, targets[:, t], file_name_base,
                                    fabcpp_options, keep_files)
                    for t in range(num_targets)]
    return feature_sets

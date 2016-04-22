import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.stats import entropy
import fs_solver import all_minimum_feature_sets as all_minfs


def diversity(patterns):
    total = 0
    for p0, p1 in combinations(patterns, 2):
        total += np.sum(x != y for x, y in zip(p0, p1))
    return total


def feature_diversity(all_features, fs_indices):
    return diversity(all_features[:, fs_indices].T)


def pattern_diversity(all_features, fs_indices):
    return diversity(all_features[:, fs_indices])


def feature_set_entropy(all_features, fs_indices):
    fs = all_features[:, fs_indices]
    counts = defaultdict(int)
    for pattern in fs:
        counts[tuple(pattern)] += 1
    return entropy(counts.values(), base=2)


def best_feature_set(features, target, method):
    ''' Takes a featureset matrix and target vector and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
    target      - <1D numpy array> of the same number of rows as features
    method      - <string> which method to use to pick best feature set.
    returns     - <1D numpy array> feature indices representing best FS
                  according to given method.'''
    if method == 'card':
        feature_sets = all_minfs(features, target)
        rand_index = np.random.randint(len(feature_sets))
        return feature_sets[rand_index, :]
    elif method == 'card>ent':
        feature_sets = all_minfs(features, target)
        entropies = [feature_set_entropy(features, fs) for fs in feature_sets]
        best_fs = numpy.argmax(entropies)
        return feature_sets[best_fs]
    elif method == 'card>fdiv':
        feature_sets = all_minfs(features, target)
        feature_diversities = [feature_diversity(features, fs)
                               for fs in feature_sets]
        best_fs = numpy.argmax(feature_diversities)
        return feature_sets[best_fs]
    elif method == 'card>pdiv':
        feature_sets = all_minfs(features, target)
        pattern_diversities = [pattern_diversity(features, fs)
                               for fs in feature_sets]
        best_fs = numpy.argmax(pattern_diversities)
        return feature_sets[best_fs]
    else:
        raise ValueError('Invalid method for feature selection: {}'.format(
            method))


def multi_target_best_feature_sets(features, targets, method):
    ''' Takes a featureset matrix and target matrix and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
    targets     - <2D numpy array> in example x feature format.
    method      - <string> which method to use to pick best feature set.
    returns     - <list of 1D numpy arrays> feature indices representing best
                  FS for each target according to given method.'''
    return [best_feature_set(features, t, method) for t in targets.T]


# import os
# import re

# regex for matching feature lines
# FEATURE_RE = re.compile('[0-9]+')

# def is_feature_set(fs, target):
#     # generator for the example index pairs to check
#     to_check = ((i1, i2) for i1, i2 in combinations(range(len(target)), 2)
#                 if target[i1] != target[i2])

#     # If all features are non-discriminating for any
#     # example pair then we do not have a feature set
#     return not any(np.array_equal(fs[e1], fs[e2]) for e1, e2 in to_check)


# def abk_file(features, target, file_name):
#     n_examples, n_features = features.shape

#     feature_numbers = np.reshape(np.arange(n_features), (n_features, 1))

#     abk_data = np.hstack((feature_numbers, features.T))

#     sample_names = '\t'.join('s' + str(i) for i in range(n_examples))

#     header = 'FEATURESINROWS\nTARGETPRESENT\nLAST\n{}\n{}\ndummy\t{}'.format(
#         n_features, n_examples, sample_names)

#     target_row = '\t' + '\t'.join(str(x) for x in target) + '\n'

#     np.savetxt(file_name, abk_data, fmt='%d', delimiter='\t',
#                header=header, footer=target_row, comments='')


# def FABCPP_cmd_line(features, target, file_name_base, options, keep_files):
#     abk_file_name = file_name_base + '.abk'
#     log_file_name = file_name_base + '.log'
#     err_file_name = file_name_base + '.err'
#     out_file_name = file_name_base + '.sol'

#     # check if files exist and raise exception

#     # write abk file
#     abk_file(features, target, abk_file_name)

#     # ensure options is a mapping type if another 'False' type is given
#     if not options:
#         options = {}

#     model = options.get('model', 6)
#     alpha = options.get('a_min', 1)
#     beta = options.get('b_min', 0)

#     cmd_string = ['fabcpp', '-i', abk_file_name, '-o', file_name_base,
#                   '-m', str(model), '-A', str(alpha), '-B', str(beta),
#                   '-O', 'max:feature_degree']

#     if model == 1:
#         cover = options.get('cover', 'alfa')
#         cmd_string += ['-y', cover]

#     # run fabcpp
#     with open(log_file_name, 'w') as log, open(err_file_name, 'w') as err:
#         check_call(cmd_string, stdout=log, stderr=err)

#     # parse output
#     with open(out_file_name) as out:
#         # pull lines from outfile which match the feature name regex
#         k_feature_set = [int(l.strip()) for l in out
#                          if FEATURE_RE.match(l.strip())]

#     if not keep_files:
#         os.remove(abk_file_name)
#         os.remove(log_file_name)
#         os.remove(err_file_name)
#         os.remove(out_file_name)

#     return np.array(k_feature_set)

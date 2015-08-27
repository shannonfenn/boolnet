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

    header = 'FEATURESINROWS\nTARGETPRESENT\nLAST\n{}\n{}\ndummy\t{}'.format(
        n_features, n_examples, '\t'.join('s' + str(i) for i in range(n_examples)))

    target_row = '\t' + '\t'.join(str(x) for x in target) + '\n'

    np.savetxt(file_name, abk_data, fmt='%d', delimiter='\t',
               header=header, footer=target_row, comments='')


def FABCPP_cmd_line(features, target, file_name_base, options):
    abk_file_name = file_name_base + '.abk'
    log_file_name = file_name_base + '.log'
    err_file_name = file_name_base + '.err'
    out_file_name = file_name_base + '.sol'

    # check if files exist and raise exception

    # write abk file
    abk_file(features, target, abk_file_name)
    # run FABCPP
    cmd_string = [
        os.path.expanduser('~/CIBMTools/FABCPP/fabcpp'),
        '-i', abk_file_name, '-o', file_name_base,
        '-m', '1', '-A', '1', '-B', '0', '-y', 'alfa',
        '-O', 'max:feature_degree']
    if options is not None:
        cmd_string += options

    with open(log_file_name, 'w') as log, open(err_file_name, 'w') as err:
        check_call(cmd_string, stdout=log, stderr=err)

    # parse output
    with open(out_file_name) as out:
        # pull out all lines from output file which match the feature name regex
        k_feature_set = [int(l.strip()) for l in out if FEATURE_RE.match(l.strip())]

        return np.array([k_feature_set])

        # TODO: Use model 5 to compute other FSs


def minimal_feature_sets(features, target, file_name_base, options):
    ''' Takes a featureset matrix and target vector and finds the
        set of all minimal featuresets.
    featureset  - assumed to be a 2D numpy array in example x feature format.
    target      - assumed to be a 1D numpy array of the same number of rows
                  as featureset
    returns     - a list of lists of feature indices representing feature
                  sets of minimal cardinality.

    NOTE: Only finds a single minFS at present'''

    return FABCPP_cmd_line(features, target, file_name_base, options)

    # check if the features form a feature set, if not then no subset can either
    # if is_feature_set(features, target):
    #     return FABCPP_cmd_line(features, target, file_name_base)
    # else:
    #     with open(file_name_base + '.err', 'w') as f:
    #         f.write('No feature set: using entire set.')
    #     return list(range(features.shape[1]))

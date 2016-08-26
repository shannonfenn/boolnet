# from boolnet.learning.feature_selection import abk_file, minimum_feature_set
import boolnet.learning.fs_solver_cplex as fss
import numpy as np
from numpy.testing import assert_array_equal
import pytest


# @fixture
# def tmpfilename():
#     random_suffix = ''.join(str(i) for i in np.random.randint(0, 9, 10))
#     return '/tmp/shantemp' + random_suffix


@pytest.fixture(params=[1, 2, 3, 4])
def instance(request):
    filename_base = 'boolnet/test/kfs_instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        feature_sets = data['feature_sets']
    return features, target, feature_sets


# def test_abk_file_generation(instance, tmpfilename):
#     features, target, _, abk_file_name = instance
#     abk_file(features, target, tmpfilename)
#     with open(tmpfilename) as f:
#         actual = f.read()
#     with open(abk_file_name) as f:
#         expected = f.read()
#     assert expected == actual


def test_min_fs(instance):
    features, target, all_expected = instance

    actual = fss.single_minimum_feature_set(features, target)

    # check the expected minfs is one of the returned
    any(np.array_equal(actual, expected) for expected in all_expected)


@pytest.mark.skip
def test_all_min_fs(instance):
    features, target, expected = instance

    actual = fss.all_minimum_feature_sets(features, target)

    # check the expected minfs is one of the returned
    np.array_equal(sorted(expected.tolist()),
                   sorted(actual.tolist()))

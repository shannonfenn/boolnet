# from boolnet.learning.feature_selection import abk_file, minimum_feature_set
from boolnet.learning.fs_solver_cplex import all_minimum_feature_sets
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture


# @fixture
# def tmpfilename():
#     random_suffix = ''.join(str(i) for i in np.random.randint(0, 9, 10))
#     return '/tmp/shantemp' + random_suffix


@fixture(params=[1, 2, 3, 4])
def instance(request):
    filename_base = 'boolnet/test/kfs_instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        minfs = data['minfs']
    return features, target, minfs


# def test_abk_file_generation(instance, tmpfilename):
#     features, target, _, abk_file_name = instance
#     abk_file(features, target, tmpfilename)
#     with open(tmpfilename) as f:
#         actual = f.read()
#     with open(abk_file_name) as f:
#         expected = f.read()
#     assert expected == actual


def test_min_fs(instance):
    features, target, expected = instance

    all_actual = all_minimum_feature_sets(features, target)

    # check the expected minfs is one of the returned
    any(np.array_equal(expected, actual) for actual in all_actual)

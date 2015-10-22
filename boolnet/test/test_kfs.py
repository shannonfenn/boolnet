from boolnet.learning.kfs import abk_file, minimum_feature_set
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture


@fixture
def tmpfilename():
    random_suffix = ''.join(str(i) for i in np.random.randint(0, 9, 10))
    return '/tmp/shantemp' + random_suffix


@fixture(params=[1, 2, 3, 4])
def instance(request):
    filename_base = 'boolnet/test/kfs_instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        minfs = data['minfs']
    abkfilename = filename_base + '.abk'

    return features, target, minfs, abkfilename


def test_abk_file_generation(instance, tmpfilename):
    features, target, _, abk_file_name = instance

    abk_file(features, target, tmpfilename)

    with open(tmpfilename) as f:
        actual = f.read()

    with open(abk_file_name) as f:
        expected = f.read()

    assert expected == actual


def test_min_fs(instance, tmpfilename):
    features, target, expected, _ = instance

    random_suffix = ''.join(str(i) for i in np.random.randint(0, 9, 10))
    tempfilename = '/tmp/shantemp' + random_suffix
    actual = minimum_feature_set(features, target, tempfilename, {}, False)

    assert_array_equal(expected, actual)

import numpy as np
from BoolNet.packing import pack_bool_matrix, unpack_bool_matrix
import pytest


@pytest.yield_fixture(params=np.random.randint(low=1, high=100, size=(100, 2)))
def shape(request):
    yield request.param


def test_packing_invertibility(shape):
    mat = np.random.randint(low=0, high=2, size=shape)
    packed_mat = pack_bool_matrix(mat)
    unpacked_mat = unpack_bool_matrix(packed_mat, mat.shape[0])
    np.testing.assert_array_equal(mat, unpacked_mat)


def test_packing_validity(shape):
    mat1 = np.random.randint(low=0, high=2, size=shape)
    mat2 = np.random.randint(low=0, high=2, size=shape)

    packed_mat1 = pack_bool_matrix(mat1)
    packed_mat2 = pack_bool_matrix(mat2)

    expected = np.logical_not(np.logical_and(mat1, mat2))

    actual_packed = np.invert(np.bitwise_and(packed_mat1, packed_mat2))

    actual = unpack_bool_matrix(actual_packed, shape[0])

    np.testing.assert_array_equal(expected, actual)


# def test_partition_examples():


# def test_update_nested():


# def test_dump():


# def test_load_datasets():

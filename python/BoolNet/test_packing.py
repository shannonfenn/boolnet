import numpy as np
from numpy.testing import assert_array_equal as assert_array_equal
from numpy.random import randint as randint
from BoolNet.packing import (packed_type, pack_bool_matrix, pack_chunk,
                             unpack_bool_matrix, unpack_bool_vector)
from pytest import fixture
from itertools import zip_longest


@fixture(params=randint(low=1, high=100, size=(100, 2)))
def shape(request):
    return request.param


@fixture
def single_matrix(shape):
    return randint(low=0, high=2, size=shape)


@fixture
def matrix_pair(shape):
    M1 = randint(low=0, high=2, size=shape)
    M2 = randint(low=0, high=2, size=shape)
    return (M1, M2)


@fixture(params=randint(low=1, high=100, size=20))
def chunk_instance(request):
    size = (64, request.param)
    return np.array(randint(low=0, high=2, size=size), dtype=np.uint8)


@fixture
def chunk_offset_instance(chunk_instance):
    cols = randint(low=3, high=10)
    offset = randint(low=0, high=cols)
    return (chunk_instance, cols, offset)


def test_pack_chunk(chunk_instance):
    _, No = chunk_instance.shape
    packed_mat = np.zeros((No, 1), dtype=packed_type)
    pack_chunk(chunk_instance, packed_mat, 0)
    for i in range(No):
        expected = pack_row(chunk_instance[:, i])[0]
        actual = packed_mat[i]
        assert expected == actual


def test_pack_chunk_offset(chunk_offset_instance):
    chunk_instance = chunk_offset_instance[0]
    cols = chunk_offset_instance[1]
    offset = chunk_offset_instance[2]
    _, No = chunk_instance.shape
    packed_mat = np.zeros((No, cols), dtype=packed_type)
    pack_chunk(chunk_instance, packed_mat, offset)
    for i in range(No):
        expected = pack_row(chunk_instance[:, i])[0]
        actual = packed_mat[i, offset]
        assert expected == actual


def test_packing_invertibility(single_matrix):
    packed_mat = pack_bool_matrix(single_matrix)
    unpacked_mat = unpack_bool_matrix(packed_mat, single_matrix.shape[0])
    assert_array_equal(single_matrix, unpacked_mat)


def test_packing_invertibility_vector(single_matrix):
    packed_mat = pack_bool_matrix(single_matrix)
    for o in range(packed_mat.shape[0]):
        unpacked_vec = unpack_bool_vector(packed_mat[o, :], single_matrix.shape[0])
        assert_array_equal(single_matrix[:, o], unpacked_vec)


def test_packing_validity(single_matrix):
    packed = pack_bool_matrix(single_matrix)
    for r in range(packed.shape[0]):
        expected = pack_row(single_matrix[:, r])
        actual = packed[r, :]
        assert_array_equal(expected, actual)


def test_size(single_matrix):
    packed = pack_bool_matrix(single_matrix)
    rows, cols = packed.shape
    Ne, No = single_matrix.shape
    assert packed.shape[0] == No
    if Ne % 64 == 0:
        assert packed.shape[1]*64 == Ne
    else:
        assert packed.shape[1]*64 == Ne + 64 - Ne % 64


def test_after_nand(matrix_pair):
    mat1, mat2 = matrix_pair
    Ne = mat1.shape[0]

    packed_mat1 = pack_bool_matrix(mat1)
    packed_mat2 = pack_bool_matrix(mat2)
    expected = np.logical_not(np.logical_and(mat1, mat2))

    actual_packed = np.invert(np.bitwise_and(packed_mat1, packed_mat2))
    actual = unpack_bool_matrix(actual_packed, Ne)

    assert_array_equal(expected, actual)


def test_after_xor(matrix_pair):
    mat1, mat2 = matrix_pair
    Ne = mat1.shape[0]

    expected = np.logical_xor(mat1, mat2)

    actual_packed = np.bitwise_xor(pack_bool_matrix(mat1), pack_bool_matrix(mat2))
    actual = unpack_bool_matrix(actual_packed, Ne)

    assert_array_equal(expected, actual)


def pack_row(row):
    return np.array(list(chunker(row, 64)), dtype=np.uint64)


def chunker(iterable, n):
    args = [iter(iterable)] * n
    for chunk in zip_longest(*args, fillvalue=0):
        yield sum(1 << i for i, b in enumerate(chunk) if b)

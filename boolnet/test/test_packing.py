import numpy as np
from numpy.testing import assert_array_equal as assert_array_equal
from numpy.random import randint as randint
import boolnet.bintools.packing as pk
from pytest import fixture, fail
from itertools import zip_longest


# @fixture(params=randint(low=2, high=100, size=(100, 2)))
# def shape(request):
#     return request.param
@fixture(params=randint(low=2, high=6, size=(100, 2)))
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


@fixture(params=randint(low=1, high=64, size=20))
def chunk_instance(request):
    No = request.param
    M = randint(low=0, high=2**(No-1), size=64)
    M = np.array(M, dtype=pk.packed_type)
    return (No, M)


@fixture
def chunk_offset_instance():
    cols = randint(low=3, high=10)
    offset = randint(low=0, high=cols)
    return cols, offset


def check_equal_transposed(M, M_p):
    No = M_p.size
    for i in range(64):
        for k in range(No):
            expected = M[i] & (np.uint64(1) << np.uint64(k)) != 0
            actual = M_p[k] & (np.uint64(1) << np.uint64(i)) != 0
            if actual != expected:
                fail('arrays differ at ({}, {}) actual={}, expected={}'.format(
                    i, k, actual, expected))


def test_pack_chunk(chunk_instance):
    No = chunk_instance[0]
    M = chunk_instance[1]
    M_p = np.zeros((No, 1), dtype=pk.packed_type)
    pk.pack_chunk(M, M_p, No, 0)

    check_equal_transposed(M, M_p[:, 0])


def test_pack_chunk_offset(chunk_instance, chunk_offset_instance):
    No = chunk_instance[0]
    M = chunk_instance[1]
    cols = chunk_offset_instance[0]
    offset = chunk_offset_instance[1]

    M_p = np.zeros((No, cols), dtype=pk.packed_type)
    pk.pack_chunk(M, M_p, No, offset)
    check_equal_transposed(M, M_p[:, offset])


def test_packing_invertibility(single_matrix):
    packed_mat = pk.pack_bool_matrix(single_matrix)
    unpacked_mat = pk.unpack_bool_matrix(packed_mat, single_matrix.shape[0])
    assert_array_equal(single_matrix, unpacked_mat)


def test_packing_invertibility_vector(single_matrix):
    packed_mat = pk.pack_bool_matrix(single_matrix)
    for o in range(packed_mat.shape[0]):
        unpacked_vec = pk.unpack_bool_vector(packed_mat[o, :], single_matrix.shape[0])
        assert_array_equal(single_matrix[:, o], unpacked_vec)


def test_packing_validity(single_matrix):
    packed = pk.pack_bool_matrix(single_matrix)
    for r in range(packed.shape[0]):
        expected = pack_row(single_matrix[:, r])
        actual = packed[r, :]
        assert_array_equal(expected, actual)


def test_size(single_matrix):
    packed = pk.pack_bool_matrix(single_matrix)
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

    packed_mat1 = pk.pack_bool_matrix(mat1)
    packed_mat2 = pk.pack_bool_matrix(mat2)
    expected = np.logical_not(np.logical_and(mat1, mat2))

    actual_packed = np.invert(np.bitwise_and(packed_mat1, packed_mat2))
    actual = pk.unpack_bool_matrix(actual_packed, Ne)

    assert_array_equal(expected, actual)


def test_after_xor(matrix_pair):
    mat1, mat2 = matrix_pair
    Ne = mat1.shape[0]

    expected = np.logical_xor(mat1, mat2)

    actual_packed = np.bitwise_xor(pk.pack_bool_matrix(mat1),
                                   pk.pack_bool_matrix(mat2))
    actual = pk.unpack_bool_matrix(actual_packed, Ne)

    assert_array_equal(expected, actual)


def pack_row(row):
    return np.array(list(chunker(row, 64)), dtype=np.uint64)


def chunker(iterable, n):
    args = [iter(iterable)] * n
    for chunk in zip_longest(*args, fillvalue=0):
        yield sum(1 << i for i, b in enumerate(chunk) if b)


def expected_partition(M, Ne, indices):
    Ms = M[indices, :]
    Mt = M[[i for i in range(Ne) if i not in indices], :]
    Msp = pk.pack_bool_matrix(Ms)
    Msp = pk.BitPackedMatrix(Msp, len(indices), 0)
    Mtp = pk.pack_bool_matrix(Mt)
    Mtp = pk.BitPackedMatrix(Mtp, Ne - len(indices), 0)
    return Msp, Mtp


def test_partition_packed(single_matrix):
    Ne = single_matrix.shape[0]

    sample_size = np.random.randint(1, Ne)
    sample = np.random.choice(Ne, size=sample_size, replace=False)
    sample.sort()

    expected_trg, expected_test = expected_partition(
        single_matrix, Ne, sample)

    packed = pk.BitPackedMatrix(
        pk.pack_bool_matrix(single_matrix), Ne, 0)

    actual_trg, actual_test = pk.partition_packed(packed, sample)

    np.testing.assert_array_equal(actual_trg, expected_trg)
    np.testing.assert_array_equal(actual_test, expected_test)


def test_sample_packed(single_matrix):
    print(single_matrix.shape)
    print(single_matrix)
    print(pk.pack_bool_matrix(single_matrix))

    Ne = single_matrix.shape[0]

    sample_size = np.random.randint(1, Ne)
    sample = np.random.choice(Ne, size=sample_size, replace=False)
    sample.sort()

    print('sample', sample)

    expected_trg, expected_test = expected_partition(
        single_matrix, Ne, sample)

    print(expected_trg)
    print(expected_test)

    packed = pk.BitPackedMatrix(
        pk.pack_bool_matrix(single_matrix), Ne, 0)

    actual_trg = pk.sample_packed(packed, sample, invert=False)
    actual_test = pk.sample_packed(packed, sample, invert=True)

    print(actual_trg)
    print(actual_test)

    np.testing.assert_array_equal(actual_trg, expected_trg)
    np.testing.assert_array_equal(actual_test, expected_test)

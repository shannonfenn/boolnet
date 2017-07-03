import math
import itertools
import numpy as np
import bitpacking.packing as pk
import boolnet.bintools.operator_iterator as opit
import boolnet.utils as utils


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def packed_from_operator(indices, Nb, No, operator, order=None, exclude=False):
    ex_iter = opit.operator_example_iterator(operator, Nb, indices, exclude)

    Ni = opit.num_operands[operator] * Nb
    Ne = len(ex_iter)

    chunks = int(math.ceil(Ne / float(pk.PACKED_SIZE_PY)))

    M = np.empty((Ni+No, chunks), dtype=pk.packed_type)

    X, Y = np.split(M, [Ni])
    pack_examples(ex_iter, X, Y)

    if order is None:
        order = np.arange(No, dtype=np.uintp)
    Y = np.array(Y[order, :])
    return utils.PackedMatrix(np.vstack((X, Y)), Ne, Ni)


def pack_examples(example_iter, X, Y):
    '''example_iter: iterator which yields pairs of unsigned ints
       X:            array_like matching pk.packed_type_t[:, :]
       Y:            array_like matching pk.packed_type_t[:, :]'''
    # check for sane dimensions
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X dim1 ({}) != Y dim1 ({})'.format(X.shape, Y.shape))
    if X.shape[1]*pk.PACKED_SIZE_PY < len(example_iter):
        raise ValueError(
            'X ({}) can\'t fit {} examples'.format(X.shape, len(example_iter)))

    X_block = np.zeros(pk.PACKED_SIZE_PY, dtype=pk.packed_type)
    Y_block = np.zeros(pk.PACKED_SIZE_PY, dtype=pk.packed_type)

    grouped = grouper(example_iter, pk.PACKED_SIZE_PY, (0, 0))
    for b, block in enumerate(grouped):
        for i, (inp, out) in enumerate(block):
            X_block[i] = inp
            Y_block[i] = out
        pk.pack_chunk(X_block, X, b)
        pk.pack_chunk(Y_block, Y, b)

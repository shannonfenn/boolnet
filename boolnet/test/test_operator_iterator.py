import operator
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
import boolnet.bintools.operator_iterator as opit


class TestIterators:
    @fixture(params=opit.num_operands.keys())
    def operator(self, request):
        return request.param

    @fixture(params=np.random.randint(3, 7, 5))
    def Nb(self, request):
        return request.param

    def random_split(self, Ni):
        max_index = 2**Ni
        num_indices = np.random.randint(1, max_index)
        in_indices = np.random.choice(max_index, size=num_indices,
                                      replace=False)
        in_indices = np.array(np.sort(in_indices), dtype=np.uint64)
        ex_indices = np.array([i for i in range(max_index)
                               if i not in in_indices])
        return in_indices, ex_indices

    def test_include_indices(self, Nb, operator):
        indices, _ = self.random_split(Nb * opit.num_operands[operator])
        expected = np.array(indices, copy=True)
        it = opit.operator_example_iterator(operator, Nb, indices, False)
        actual = np.array(list(it))[:, 0]

        assert_array_equal(expected, actual)

    def test_exclude_indices(self, Nb, operator):
        indices, expected = self.random_split(Nb * opit.num_operands[operator])
        it = opit.operator_example_iterator(operator, Nb, indices, True)
        actual = np.array(list(it))[:, 0]

        assert_array_equal(expected, actual)


class TestExampleIteratorFactory:
    @fixture(params=[
        ('and', operator.and_),
        ('or', operator.or_),
        ('add', operator.add),
        ('sub', operator.sub),
        ('mul', operator.mul)
    ])
    def binary_op(self, request):
        return request.param

    @fixture(params=[
        ('zero', lambda x, m: 0),
        ('unary_or', lambda x, m: int(x > 0)),
        ('unary_and', lambda x, m: int(x == m - 1))
    ])
    def unary_op(self, request):
        return request.param

    @fixture(params=list(range(1, 7)))
    def operand_width(self, request):
        return request.param

    def test_operator_factory_include(self, binary_op, operand_width):
        name, func = binary_op
        Nb = int(operand_width)
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        it = opit.operator_example_iterator(name, Nb, indices, False)

        for i, (inp, tgt) in zip(indices, it):
            assert i == inp
            expected_out = func(int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_operator_factory_exclude(self, binary_op, operand_width):
        name, func = binary_op
        Nb = operand_width
        upper = 2**Nb
        N = 2**(2*Nb)
        Ne = np.random.randint(min(100, N))
        ex_indices = np.random.choice(N, Ne, replace=False)
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(N) if i not in ex_indices)

        it = opit.operator_example_iterator(name, Nb, ex_indices, True)

        for i, (inp, tgt) in zip(indices, it):
            assert i == inp
            expected_out = func(int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_unary_operator_factory_include(self, unary_op, operand_width):
        name, func = unary_op
        Nb = int(operand_width)
        N = 2**Nb
        Ne = np.random.randint(min(100, N))
        indices = np.random.choice(N, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        it = opit.operator_example_iterator(name, Nb, indices, False)

        for i, (inp, tgt) in zip(indices, it):
            assert i == inp
            expected_out = func(i, N)
            assert expected_out == tgt

    def test_unary_operator_factory_exclude(self, unary_op, operand_width):
        name, func = unary_op
        Nb = operand_width
        N = 2**Nb
        Ne = np.random.randint(min(100, N))
        ex_indices = np.sort(np.random.choice(N, Ne, replace=False))
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(N) if i not in ex_indices)

        it = opit.operator_example_iterator(name, Nb, ex_indices, True)

        for i, (inp, tgt) in zip(indices, it):
            assert i == inp
            expected_out = func(i, N)
            assert expected_out == tgt

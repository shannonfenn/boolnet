import operator
import os.path
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
from boolnet.bintools.packing import packed_type, pack_bool_matrix
import boolnet.bintools.operator_iterator as op_it
from boolnet.bintools.example_generator import OperatorExampleIteratorFactory, PackedExampleGenerator


class TestIterators:
    @fixture(params=[
        op_it.ZeroIncludeIterator,
        op_it.UnaryANDIncludeIterator,
        op_it.UnaryORIncludeIterator,
        op_it.ANDIncludeIterator,
        op_it.ORIncludeIterator,
        op_it.AddIncludeIterator,
        op_it.SubIncludeIterator,
        op_it.MulIncludeIterator])
    def include_iterator(self, request):
        return request.param

    @fixture(params=[
        op_it.ZeroExcludeIterator,
        op_it.UnaryANDExcludeIterator,
        op_it.UnaryORExcludeIterator,
        op_it.ANDExcludeIterator,
        op_it.ORExcludeIterator,
        op_it.AddExcludeIterator,
        op_it.SubExcludeIterator,
        op_it.MulExcludeIterator])
    def exclude_iterator(self, request):
        return request.param

    @fixture(params=np.random.randint(2, 100, 5))
    def index_harness(self, request):
        max_index = request.param
        num_indices = np.random.randint(max_index-1, max_index)
        in_indices = np.random.choice(max_index, size=num_indices, replace=False)
        in_indices = np.array(np.sort(in_indices), dtype=np.uint64)
        ex_indices = np.array([i for i in range(max_index) if i not in in_indices])
        return (in_indices, ex_indices, max_index)

    def test_include_indices(self, include_iterator, index_harness):
        indices, _, _ = index_harness
        expected = np.array(indices, copy=True)
        actual = np.array(list(include_iterator(indices, 2)))[:, 0]

        assert_array_equal(expected, actual)

    def test_exclude_indices(self, exclude_iterator, index_harness):
        indices, expected, max_index = index_harness
        print(expected.size)
        actual = np.array(list(exclude_iterator(indices, 2, max_index)))[:, 0]

        assert_array_equal(expected, actual)


class TestExampleIteratorFactory:
    @fixture(params=[
        (op_it.AND, operator.__and__),
        (op_it.OR, operator.__or__),
        (op_it.ADD, operator.add),
        (op_it.SUB, operator.sub),
        (op_it.MUL, operator.mul)
    ])
    def binary_op(self, request):
        return request.param

    @fixture(params=[
        (op_it.ZERO, lambda x, m: 0),
        (op_it.UNARY_OR, lambda x, m: int(x > 0)),
        (op_it.UNARY_AND, lambda x, m: int(x == m - 1))
    ])
    def unary_op(self, request):
        return request.param

    @fixture(params=list(range(1, 7)))
    def operand_width(self, request):
        return request.param

    def test_operator_factory_include(self, binary_op, operand_width):
        operator_id, operator_function = binary_op
        Nb = int(operand_width)
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = OperatorExampleIteratorFactory(indices, Nb, operator_id)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = operator_function(int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_operator_factory_exclude(self, binary_op, operand_width):
        Nb = operand_width
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        # ex_indices = np.sort(np.random.choice(max_indices, Ne, replace=False))
        ex_indices = np.random.choice(max_indices, Ne, replace=False)
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(max_indices) if i not in ex_indices)

        factory = OperatorExampleIteratorFactory(ex_indices, Nb, binary_op[0], max_indices)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = binary_op[1](int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_unary_operator_factory_include(self, unary_op, operand_width):
        Nb = int(operand_width)
        max_indices = 2**Nb
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = OperatorExampleIteratorFactory(indices, Nb, unary_op[0])

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, max_indices)
            assert expected_out == tgt

    def test_unary_operator_factory_exclude(self, unary_op, operand_width):
        Nb = operand_width
        max_indices = 2**Nb
        Ne = np.random.randint(min(100, max_indices))
        ex_indices = np.sort(np.random.choice(max_indices, Ne, replace=False))
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(max_indices) if i not in ex_indices)

        factory = OperatorExampleIteratorFactory(ex_indices, Nb, unary_op[0], max_indices)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, max_indices)
            assert expected_out == tgt


class TestExampleGenerator:
    op_map = {'add': op_it.ADD, 'sub': op_it.SUB, 'mul': op_it.MUL}
    cache = dict()

    @fixture(params=['add4.npz', 'add8.npz', 'sub4.npz', 'sub8.npz',
                     'mul2.npz', 'mul3.npz', 'mul4.npz', 'mul6.npz'])
    def file_func_inst(self, request, test_location):
        return test_location, request.param

    @fixture(params=[True, False])
    def include(self, request):
        return request.param

    def load_file_func_instance(self, instance):
        location, name = instance
        fname = os.path.join(location, 'functions', name)
        if fname not in self.cache:
            with np.load(fname) as data:
                inp = data['input_matrix']
                tgt = data['target_matrix']
            self.cache[fname] = (inp, tgt)
        inp = np.array(self.cache[fname][0], copy=True)
        tgt = np.array(self.cache[fname][1], copy=True)
        return self.op_map[name[:3]], inp, tgt

    def build_generator_instance(self, instance, include):
        op, inp, tgt = self.load_file_func_instance(instance)

        Ne, Ni = inp.shape
        _, No = tgt.shape

        indices = np.random.choice(Ne, min(100, Ne), replace=False)
        indices = np.sort(indices)
        inp = inp[indices, :]
        tgt = tgt[indices, :]
        inp_p = pack_bool_matrix(inp)
        tgt_p = pack_bool_matrix(tgt)
        if not include:
            indices = [i for i in range(Ne) if i not in indices]
        indices = np.array(indices, dtype=packed_type)

        factory = OperatorExampleIteratorFactory(indices, Ni//2, op, 0 if include else Ne)

        gen = PackedExampleGenerator(factory, No)
        return (gen, inp_p, tgt_p)

    @fixture(params=[0, np.random.rand(), 1])
    def block_fraction(self, request):
        return request.param

    def test_packed_generation(self, file_func_inst, include, block_fraction):
        generator_instance = self.build_generator_instance(file_func_inst, include)
        generator, expected_inp, expected_tgt = generator_instance

        actual_inp = np.zeros_like(expected_inp)
        actual_tgt = np.zeros_like(expected_tgt)
        Ni, cols = expected_inp.shape
        No, _ = expected_tgt.shape
        block_width = max(1, cols * block_fraction)

        i = 0
        while generator:
            generator.next_examples(actual_inp[:, i: i + block_width],
                                    actual_tgt[:, i: i + block_width])
            i += block_width
        assert_array_equal(expected_inp, actual_inp)
        assert_array_equal(expected_tgt, actual_tgt)

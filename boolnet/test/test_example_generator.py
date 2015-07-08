import operator
import os.path
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
from boolnet.bintools.packing import packed_type, pack_bool_matrix
from boolnet.bintools.operator_iterator import ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL
from boolnet.bintools.example_generator import OperatorExampleFactory, PackedExampleGenerator


class TestExampleFactory:
    @fixture(params=[
        (AND, operator.__and__),
        (OR, operator.__or__),
        (ADD, operator.add),
        (SUB, operator.sub),
        (MUL, operator.mul)
    ])
    def binary_op(self, request):
        return request.param

    @fixture(params=[
        (ZERO, lambda x, m: 0),
        (UNARY_OR, lambda x, m: int(x > 0)),
        (UNARY_AND, lambda x, m: int(x == m - 1))
    ])
    def unary_op(self, request):
        return request.param

    @fixture(params=list(range(1, 7)))
    def operand_width(self, request):
        return request.param

    def test_operator_factory_include(self, binary_op, operand_width):
        Nb = int(operand_width)
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = OperatorExampleFactory(indices, Nb, binary_op[0])

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = binary_op[1](int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_operator_factory_exclude(self, binary_op, operand_width):
        Nb = operand_width
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        ex_indices = np.sort(np.random.choice(max_indices, Ne, replace=False))
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(max_indices) if i not in ex_indices)

        factory = OperatorExampleFactory(ex_indices, Nb, binary_op[0], max_indices)

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

        factory = OperatorExampleFactory(indices, Nb, unary_op[0])

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

        factory = OperatorExampleFactory(ex_indices, Nb, unary_op[0], max_indices)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, max_indices)
            assert expected_out == tgt


class TestExampleGenerator:
    op_map = {'add': ADD, 'sub': SUB, 'mul': MUL}
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

        factory = OperatorExampleFactory(indices, Ni//2, op,
                                         0 if include else Ne)

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

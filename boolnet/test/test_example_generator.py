import operator
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_almost_equal as assert_array_almost_equal
from boolnet.bintools.packing import packed_type, pack_bool_matrix
from boolnet.bintools.example_generator import OperatorExampleFactory


class TestExampleGenerator:
    @fixture(params=[
        dict(name='add4', operator=operator.add),
        dict(name='add8', operator=operator.add),
        dict(name='sub4', operator=operator.sub),
        dict(name='sub8', operator=operator.sub),
        dict(name='mul2', operator=operator.mul),
        dict(name='mul4', operator=operator.mul),
        dict(name='mul6', operator=operator.mul),
        dict(name='mul2-overflow', operator=operator.mul),
        dict(name='mul4-overflow', operator=operator.mul),
        dict(name='mul6-overflow', operator=operator.mul)
    ])
    def file_func_inst(self, request, test_location):
        d = request.param
        d['name'] = test_location + d['name']
        return d

    @fixture(params=[operator.add, operator.sub, operator.mul])
    def binary_operator(self, request):
        return request.param

    @fixture(params=list(range(1, 7)))
    def operand_width(self, request):
        return request.param

    def test_operator_factory_include(self, binary_operator, operand_width):
        Nb = operand_width
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(max_indices)
        indices = np.random.choice(max_indices, Ne, replace=False).tolist()
        index_factory = lambda: (i for i in indices)

        factory = OperatorExampleFactory(index_factory, binary_operator, Ne, Nb, True)
        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            assert binary_operator(i // upper, i % upper) == tgt

    def test_operator_factory_exclude(self, binary_operator, operand_width):
        Nb = operand_width
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(max_indices)
        ex_indices = np.sort(np.random.choice(max_indices, Ne, replace=False)).tolist()
        index_factory = lambda: (i for i in ex_indices)
        indices = (i for i in range(max_indices) if i not in ex_indices)

        factory = OperatorExampleFactory(index_factory, binary_operator, Ne, Nb, False)

        # print(Nb, upper, Ne)
        # print(ex_indices)
        # print(max_indices)
        for i, (inp, tgt) in zip(indices, iter(factory)):
            # print(i, inp, tgt)
            assert i == inp
            assert binary_operator(i // upper, i % upper) == tgt

    # @fixture
    # def include_indices(file_func_inst):
    #     op = file_func_inst['operator']
    #     with np.load(file_func_inst['name'] + '.npz') as data:
    #         inp = data['target_matrix']
    #         tgt = data['input_matrix']
    #     Ne, Ni = inp.shape
    #     _, No = tgt.shape
    #     inp_p = pack_bool_matrix(inp)
    #     tgt_p = pack_bool_matrix(tgt)

    #     indices = np.random.choice(Ne, 100)

    #     factory = OperatorExampleFactory(indices, op, Ne, No, False)
    #     return (factory, inp_p, tgt_p)

    # @fixture
    # def exclude_instance(file_func_inst):
    #     op = file_func_inst['operator']
    #     with np.load(file_func_inst['name'] + '.npz') as data:
    #         inp = data['target_matrix']
    #         tgt = data['input_matrix']
    #     Ne, Ni = inp.shape
    #     _, No = tgt.shape
    #     inp_p = pack_bool_matrix(inp)
    #     tgt_p = pack_bool_matrix(tgt)

    #     indices = np.random.choice(Ne, Ne // 2)

    #     factory = OperatorExampleFactory(indices, op, Ne, No, True)
    #     return (factory, inp_p, tgt_p)

    # @fixture
    # def include_instance(binary_function):
    #     op = binary_function['operator']
    #     with np.load(binary_function['name'] + '.npz') as data:
    #         inp = data['target_matrix']
    #         tgt = data['input_matrix']
    #     Ne, Ni = inp.shape
    #     _, No = tgt.shape
    #     inp_p = pack_bool_matrix(inp)
    #     tgt_p = pack_bool_matrix(tgt)

    #     indices = np.random.choice(Ne, 100)

    #     factory = OperatorExampleFactory(indices, op, Ne, No, False)
    #     return (factory, inp_p, tgt_p)

    # @fixture
    # def exclude_instance(binary_function):
    #     op = binary_function['operator']
    #     with np.load(binary_function['name'] + '.npz') as data:
    #         inp = data['target_matrix']
    #         tgt = data['input_matrix']
    #     Ne, Ni = inp.shape
    #     _, No = tgt.shape
    #     inp_p = pack_bool_matrix(inp)
    #     tgt_p = pack_bool_matrix(tgt)

    #     indices = np.random.choice(Ne, Ne // 2)

    #     factory = OperatorExampleFactory(indices, op, Ne, No, True)
    #     return (factory, inp_p, tgt_p)

    # def test_include_factory_input(self, include_instance):
    #     factory, expected, _ = include_instance

    #     actual = np.zeros_like(expected)


        
    #     example_factory = OperatorExampleFactory()

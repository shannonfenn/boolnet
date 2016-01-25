import yaml
import glob
import numpy as np
from copy import copy, deepcopy
from collections import namedtuple
from pytest import mark, raises, fixture

from boolnet.bintools.packing import (
    pack_bool_matrix, unpack_bool_matrix, generate_end_mask, BitPackedMatrix,
    partition_packed)
from boolnet.bintools.packing import PACKED_SIZE_PY as PACKED_SIZE
from boolnet.bintools.functions import (
    function_name, all_functions, function_from_name)
from boolnet.bintools.operator_iterator import (
    ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL,
    OpExampleIterFactory)
from boolnet.bintools.example_generator import (
    packed_from_operator, PackedExampleGenerator)
from boolnet.learning.networkstate import (
    StandardBNState, ChainedBNState,
    standard_from_operator, chained_from_operator)


TEST_NETWORKS = glob.glob('boolnet/test/networks/*.yaml')


operator_map = {
    'and': AND, 'or': OR,
    'add': ADD, 'sub': SUB, 'mul': MUL,
    'zero': ZERO, 'unary_and': UNARY_AND, 'unary_or': UNARY_OR
}


# ############# General helpers ################# #
def to_binary(value, num_bits):
    return [int(i) for i in '{:0{w}b}'.format(value, w=num_bits)][::-1]


def all_possible_inputs(num_bits):
    return pack_bool_matrix(np.array(
        [to_binary(i, num_bits) for i in range(2**num_bits)],
        dtype=np.byte))


def packed_zeros(shape):
    return pack_bool_matrix(np.zeros(shape, dtype=np.uint8))


# ############ Helpers for fixtures ############# #
def harnesses_with_property(bool_property_name):
    for name in TEST_NETWORKS:
        with open(name) as f:
            test = yaml.safe_load(f)
            if test[bool_property_name]:
                yield name


HARNESS_CACHE = dict()


def harness_to_fixture(fname, state_type):
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            test = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = test

    test = deepcopy(HARNESS_CACHE[fname])

    if state_type == 'standard':
        return standard_harness_to_fixture(test)
    elif state_type == 'chained':
        return chained_harness_to_fixture(test)


def standard_harness_to_fixture(test):
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    samples = np.array(test['samples'], np.uint32)

    # add non-existant sub-dictionaries
    test['input matrix'] = {}
    test['output matrix'] = {}

    target = np.array(test['target matrix']['full'], dtype=np.uint8)
    activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    error = np.array(test['error matrix']['full'], dtype=np.uint8)
    inputs = activation[:, :Ni]
    test['target matrix']['full'] = target
    test['input matrix']['full'] = inputs
    test['output matrix']['full'] = activation[:, -No:]
    test['activation matrix']['full'] = activation
    test['error matrix']['full'] = error

    # add sample version of expectations to test
    test['target matrix']['sample'] = target[samples]
    test['input matrix']['sample'] = inputs[samples]
    test['output matrix']['sample'] = activation[samples][:, -No:]
    test['activation matrix']['sample'] = activation[samples]
    test['error matrix']['sample'] = error[samples]

    # Test sample version
    samples_t = np.array([i for i in range(inputs.shape[0])
                          if i not in samples])

    target_t = target[samples_t]
    inputs_t = inputs[samples_t]
    activation_t = activation[samples_t]
    # add test version of expectations to test
    test['target matrix']['test'] = target_t
    test['input matrix']['test'] = inputs_t
    test['output matrix']['test'] = activation_t[:, -No:]
    test['activation matrix']['test'] = activation_t
    test['error matrix']['test'] = error[samples_t]

    # generate sample versions
    Mf = BitPackedMatrix(
        np.vstack((pack_bool_matrix(inputs),
                   pack_bool_matrix(target))),
        Ne=inputs.shape[0], Ni=Ni)

    # This is wrong, need to use packed sampling methods
    # but they will handle Ne which is good
    Ms, Mt = partition_packed(Mf, samples)

    # add states to test
    test['state'] = {
        'full': StandardBNState(gates, Mf),
        'sample': StandardBNState(gates, Ms),
        'test': StandardBNState(gates, Mt)
    }

    return test


def chained_harness_to_fixture(test):
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)

    indices_f = np.arange(2**Ni, dtype=np.uint32)
    indices_s = np.array(test['samples'], dtype=np.uint32)

    op = operator_map[test['target function']]

    if test['target function'] in ['zero', 'unary_and', 'unary_or']:
        Nb = Ni
    else:
        Nb = Ni // 2

    iter_factory_f = OpExampleIterFactory(indices_f, Nb, op, exclude=False)
    iter_factory_s = OpExampleIterFactory(indices_s, Nb, op, exclude=False)
    iter_factory_t = OpExampleIterFactory(indices_s, Nb, op, exclude=True)

    generator_f = PackedExampleGenerator(iter_factory_f, No)
    generator_s = PackedExampleGenerator(iter_factory_s, No)
    generator_t = PackedExampleGenerator(iter_factory_t, No)

    Ne_f = 2**Ni
    Ne_s = indices_s.size
    Ne_t = Ne_f - Ne_s

    window_size_f = np.random.randint(1, max(2, Ne_f // PACKED_SIZE))
    window_size_s = np.random.randint(1, max(2, Ne_s // PACKED_SIZE))
    window_size_t = np.random.randint(1, max(2, Ne_t // PACKED_SIZE))

    # add states to test
    test['state'] = {
        'full': ChainedBNState(gates, generator_f, window_size_f),
        'sample': ChainedBNState(gates, generator_s, window_size_s),
        'test': ChainedBNState(gates, generator_t, window_size_t),
    }

    return test


@fixture(params=['sample', 'full'])
def sample_type(request):
    return request.param


@fixture(params=['standard', 'chained'])
def state_type(request):
    return request.param


@fixture(params=TEST_NETWORKS)
def chained_state(request):
    return harness_to_fixture(request.param, 'chained')


@fixture(params=TEST_NETWORKS)
def state_harness(request, state_type):
    return harness_to_fixture(request.param, state_type)


@fixture(params=TEST_NETWORKS)
def state_params(request):
    fname = request.param
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            harness = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = harness
    # copy to ensure harness cache remains correct
    test = deepcopy(HARNESS_CACHE[fname])

    Ni = test['Ni']
    if test['target function'] in ['zero', 'unary_and', 'unary_or']:
        test['Nb'] = Ni
    else:
        test['Nb'] = Ni // 2
    test['operator'] = operator_map[test['target function']]

    indices_s = np.array(test['samples'], dtype=np.uint32)
    indices_f = np.arange(2**Ni, dtype=np.uint32)

    test['exclude'] = {'full': False, 'sample': False, 'test': True}

    # add generated params to test
    test['indices'] = {
        'sample': indices_s,
        'full': indices_f,
        'test': indices_s
    }

    Ne_f = 2**Ni
    Ne_s = indices_s.size
    Ne_t = Ne_f - Ne_s

    test['window_size'] = {
        'sample': np.random.randint(1, max(2, Ne_s // PACKED_SIZE)),
        'full': np.random.randint(1, max(2, Ne_f // PACKED_SIZE)),
        'test': np.random.randint(1, max(2, Ne_t // PACKED_SIZE))
    }

    return test


@fixture(params=list(harnesses_with_property('invariant under single move')))
def single_move_invariant(request):
    return harness_to_fixture(request.param, 'standard')


@fixture(params=list(harnesses_with_property('invariant under multiple moves')))
def multiple_move_invariant(request):
    return harness_to_fixture(request.param, 'standard')


MoveAndExpected = namedtuple('MoveAnExpected', ['move', 'expected'])


# ################### Functionality Testing ################### #
class TestStandard:

    # ##################### HELPERS ##################### #
    def build_instance(self, instance_dict, sample_type, field):
        state = instance_dict['state'][sample_type]
        expected = np.array(instance_dict[field][sample_type], dtype=np.uint8)
        return state, expected, lambda mat: unpack_bool_matrix(mat, state.Ne)

    def output_different_helper(self, instance):
        state, _, eval_func = self.build_instance(
            instance, 'full', 'error matrix')
        for k in range(10):
            old_error = eval_func(state.error_matrix)
            state.move_to_random_neighbour()
            assert not np.array_equal(eval_func(state.error_matrix), old_error)
            state.revert_move()

    # ##################### FIXTURES ##################### #
    @fixture(params=TEST_NETWORKS)
    def standard_state(self, request):
        return harness_to_fixture(request.param, 'standard')

    @fixture
    def single_layer_zero(self):
        instance = harness_to_fixture(
            'boolnet/test/networks/single_layer_zero.yaml', 'standard')

        instance = copy(instance)

        test_case = instance['multiple_moves_test_case']

        updated_test_case = []
        for step in test_case:
            move = step['move']
            expected = np.array(step['expected'], dtype=np.uint8)
            updated_test_case.append(
                MoveAndExpected(move=move, expected=expected))

        instance['multiple_moves_test_case'] = updated_test_case
        return instance

    # ################### Exception Testing ################### #
    @mark.parametrize("Ni, Tshape, Ne", [
        (1, (2, 4), 4),     # net.No != #tgts
        (2, (1, 4), 4),     # net.Ni != #inps
        (2, (2, 4), 4)      # both
    ])
    def test_construction_exceptions(self, Ni, Tshape, Ne):
        M = BitPackedMatrix(np.vstack((
            all_possible_inputs(Ni),
            packed_zeros(Tshape))), Ne, Ni)
        with raises(ValueError):
            StandardBNState([(0, 1, 3)], M)

    # ################### Functionality Testing ################### #
    def test_input_matrix(self, standard_state, sample_type):
        state, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'input matrix')
        actual = eval_func(state.input_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_target_matrix(self, standard_state, sample_type):
        state, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'target matrix')
        actual = eval_func(state.target_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_output_matrix(self, standard_state, sample_type):
        state, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'output matrix')
        actual = eval_func(state.output_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_activation_matrix(self, standard_state, sample_type):
        state, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'activation matrix')
        actual = eval_func(state.activation_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_error_matrix(self, standard_state, sample_type):
        state, expected, eval_func = self.build_instance(
            standard_state, sample_type, 'error matrix')
        actual = eval_func(state.error_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_single_move_output_different(self, single_move_invariant):
        self.output_different_helper(single_move_invariant)

    def test_multiple_move_output_different(self, multiple_move_invariant):
        self.output_different_helper(multiple_move_invariant)

    def test_move_with_initial_evaluation(self, single_layer_zero):
        state, expected, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')

        actual = eval_func(state.error_matrix)
        np.testing.assert_array_equal(expected, actual)

        test_case = single_layer_zero['multiple_moves_test_case'][4]
        expected = test_case.expected

        state.apply_move(test_case.move)
        actual = eval_func(state.error_matrix)
        np.testing.assert_array_equal(expected, actual)

    def test_multiple_moves_error_matrix(self, single_layer_zero):
        state, _, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, expected in test_case:
            state.apply_move(move)
            actual = eval_func(state.error_matrix)
            np.testing.assert_array_equal(expected, actual)

    def test_multiple_reverts_error_matrix(self, single_layer_zero):
        state, _, eval_func = self.build_instance(
            single_layer_zero, 'full', 'error matrix')

        test_case = single_layer_zero['multiple_moves_test_case']

        for move, _ in test_case:
            state.apply_move(move)

        for _, expected in reversed(test_case):
            actual = eval_func(state.error_matrix)
            np.testing.assert_array_equal(expected, actual)
            state.revert_move()

    def test_pre_evaluated_network(self, standard_state):
        state_s, expected, eval_func_s = self.build_instance(
            standard_state, 'sample', 'activation matrix')

        for i in range(10):
            state_s.move_to_random_neighbour()
            state_s.evaluate()
        state_s.revert_all_moves()

        # check sample state is still giving original results
        actual = eval_func_s(state_s.activation_matrix)
        np.testing.assert_array_equal(expected, actual)

        state_f, expected, eval_func_f = self.build_instance(
            standard_state, 'full', 'activation matrix')

        # check full state is still giving original results
        state_f.set_gates(state_s.gates)
        actual = eval_func_f(state_f.activation_matrix)

        np.testing.assert_array_equal(expected, actual)


class TestBoth:

    def build_from_params(self, params, eval_type, sample_type):
        if eval_type == 'standard':
            return standard_from_operator(
                gates=params['gates'],
                indices=params['indices'][sample_type],
                Nb=params['Nb'], No=params['No'],
                operator=params['operator'],
                exclude=params['exclude'][sample_type]
            )
        elif eval_type == 'chained':
            return chained_from_operator(
                gates=params['gates'],
                indices=params['indices'][sample_type],
                Nb=params['Nb'], No=params['No'],
                operator=params['operator'],
                window_size=params['window_size'][sample_type],
                exclude=params['exclude'][sample_type]
            )

    # ################### Exception Testing ################### #
    def test_function_not_added(self, chained_state, function, sample_type):
        with raises(ValueError):
            chained_state['state'][sample_type].function_value(function)

    # ################### Functionality Testing ################### #
    def test_from_operator_combined_attributes(self, state_params, state_type, sample_type):
        state = self.build_from_params(state_params, state_type, sample_type)

        assert state.Ni == state_params['Ni']
        assert state.No == state_params['No']
        assert state.Ng == len(state_params['gates'])
        assert state.zero_mask == generate_end_mask(state.Ne)

    def test_from_operator_func_value(self, state_params, state_type, function, sample_type):
        expected = state_params['function value'][sample_type][function_name(function)]
        state = self.build_from_params(state_params, state_type, sample_type)
        state.add_function(function)
        actual = state.function_value(function)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_function_value(self, state_harness, function, sample_type):
        state = state_harness['state'][sample_type]
        expected = state_harness['function value'][sample_type][function_name(function)]
        state.add_function(function)
        actual = state.function_value(function)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_multiple_function_values(self, state_harness, sample_type):
        state = state_harness['state'][sample_type]
        for function in all_functions():
            state.add_function(function)
        for function in all_functions():
            expected = state_harness['function value'][sample_type][function_name(function)]
            actual = state.function_value(function)
            np.testing.assert_array_almost_equal(expected, actual)


ind = [[125, 198, 54, 40, 228, 20, 63, 124, 36, 81, 64, 141, 61, 149, 166, 2, 100, 38, 45, 18, 183, 220, 213, 78, 150, 37, 146, 207, 173, 101, 127, 27, 105, 204, 212, 185, 57, 107, 129, 248, 175, 47, 180, 244, 32, 164, 82, 79],
       [125, 198, 54, 40, 228, 20, 63, 124, 36, 81, 64, 141, 61, 149, 166, 2, 100, 38, 45, 18, 183, 220, 213, 78, 150, 37, 146, 207, 173, 101, 127, 27, 105, 204, 212, 185, 57, 107, 129, 248, 175, 47, 180, 244, 32, 164, 82, 79],
       [181, 105, 138, 48, 41, 73, 24, 101, 132, 67, 22, 154, 125, 66, 3, 49, 26, 63, 57, 99, 222, 33, 185, 80, 204, 219, 123, 212, 228, 51, 107, 176, 6, 201, 28, 25, 148, 74, 58, 146, 70, 223, 71, 20, 120, 217, 18, 147],
       [10, 221, 9, 15, 244, 149, 60, 219, 237, 68, 66, 58, 50, 103, 24, 193, 65, 230, 119, 69, 81, 134, 253, 43, 186, 82, 250, 98, 12, 243, 72, 30, 29, 179, 102, 104, 14, 209, 2, 112, 182, 77, 99, 160, 224, 226, 61, 76],
       [118, 234, 193, 48, 176, 214, 83, 163, 110, 179, 212, 60, 55, 241, 61, 247, 127, 52, 219, 131, 81, 126, 130, 157, 5, 17, 170, 18, 43, 0, 59, 99, 153, 218, 142, 90, 253, 177, 36, 122, 19, 120, 181, 223, 161, 116, 146, 184],
       [118, 234, 193, 48, 176, 214, 83, 163, 110, 179, 212, 60, 55, 241, 61, 247, 127, 52, 219, 131, 81, 126, 130, 157, 5, 17, 170, 18, 43, 0, 59, 99, 153, 218, 142, 90, 253, 177, 36, 122, 19, 120, 181, 223, 161, 116, 146, 184],
       [82, 6, 79, 47, 163, 124, 216, 93, 63, 125, 83, 214, 217, 69, 188, 34, 38, 176, 164, 223, 103, 170, 132, 139, 213, 105, 112, 118, 3, 39, 111, 144, 90, 130, 24, 232, 230, 134, 114, 89, 15, 128, 189, 99, 37, 56, 129, 95],
       [181, 105, 138, 48, 41, 73, 24, 101, 132, 67, 22, 154, 125, 66, 3, 49, 26, 63, 57, 99, 222, 33, 185, 80, 204, 219, 123, 212, 228, 51, 107, 176, 6, 201, 28, 25, 148, 74, 58, 146, 70, 223, 71, 20, 120, 217, 18, 147],
       [10, 221, 9, 15, 244, 149, 60, 219, 237, 68, 66, 58, 50, 103, 24, 193, 65, 230, 119, 69, 81, 134, 253, 43, 186, 82, 250, 98, 12, 243, 72, 30, 29, 179, 102, 104, 14, 209, 2, 112, 182, 77, 99, 160, 224, 226, 61, 76],
       [82, 6, 79, 47, 163, 124, 216, 93, 63, 125, 83, 214, 217, 69, 188, 34, 38, 176, 164, 223, 103, 170, 132, 139, 213, 105, 112, 118, 3, 39, 111, 144, 90, 130, 24, 232, 230, 134, 114, 89, 15, 128, 189, 99, 37, 56, 129, 95]]
G = [[[0, 4, 7], [7, 4, 7], [9, 5, 7], [8, 7, 7], [7, 0, 7], [8, 0, 7], [3, 3, 7], [10, 0, 7], [7, 10, 7], [0, 11, 7], [15, 15, 7], [11, 0, 7], [15, 9, 7], [3, 16, 7], [4, 17, 7], [4, 16, 7], [1, 17, 7], [4, 20, 7], [11, 10, 7], [9, 13, 7], [19, 10, 7], [25, 1, 7], [1, 17, 7], [8, 5, 7], [30, 29, 7], [15, 4, 7], [15, 6, 7], [14, 28, 7], [2, 7, 7], [10, 25, 7], [15, 22, 7], [35, 31, 7], [32, 31, 7], [1, 4, 7], [24, 37, 7], [27, 11, 7], [10, 7, 7], [41, 22, 7], [1, 16, 7], [46, 39, 7], [6, 41, 7], [5, 31, 7], [44, 4, 7], [49, 48, 7], [18, 2, 7], [2, 51, 7], [42, 4, 7], [44, 34, 7], [33, 46, 7], [31, 15, 7], [49, 54, 7], [43, 29, 7], [21, 49, 7], [53, 32, 7], [12, 20, 7], [23, 4, 7], [53, 61, 7], [58, 6, 7], [5, 57, 7], [35, 42, 7], [46, 46, 7], [47, 50, 7], [66, 37, 7], [50, 50, 7], [5, 59, 7], [33, 30, 7], [42, 26, 7], [60, 69, 7], [22, 4, 7], [7, 61, 7], [66, 1, 7], [36, 75, 7], [32, 36, 7], [55, 73, 7], [53, 79, 7], [4, 49, 7], [42, 66, 7], [26, 35, 7], [1, 31, 7], [57, 58, 7], [22, 13, 7], [47, 40, 7], [64, 65, 7], [82, 74, 7]],
     [[0, 4, 7], [7, 6, 7], [7, 0, 7], [7, 5, 7], [7, 8, 7], [8, 0, 7], [0, 4, 7], [10, 4, 7], [14, 8, 7], [12, 0, 7], [8, 6, 7], [12, 13, 7], [17, 3, 7], [16, 13, 7], [21, 4, 7], [9, 12, 7], [9, 23, 7], [17, 13, 7], [6, 0, 7], [3, 25, 7], [1, 5, 7], [13, 5, 7], [14, 5, 7], [19, 20, 7], [11, 14, 7], [23, 16, 7], [28, 29, 7], [18, 9, 7], [1, 13, 7], [2, 0, 7], [24, 36, 7], [5, 9, 7], [11, 4, 7], [22, 25, 7], [17, 6, 7], [36, 34, 7], [7, 20, 7], [15, 33, 7], [39, 38, 7], [26, 42, 7], [23, 22, 7], [47, 2, 7], [8, 14, 7], [43, 37, 7], [16, 2, 7], [7, 29, 7], [13, 2, 7], [54, 30, 7], [1, 28, 7], [50, 46, 7], [54, 34, 7], [56, 28, 7], [49, 21, 7], [9, 23, 7], [20, 38, 7], [34, 4, 7], [13, 38, 7], [3, 58, 7], [58, 49, 7], [2, 43, 7], [42, 17, 7], [55, 48, 7], [5, 9, 7], [23, 50, 7], [32, 58, 7], [72, 40, 7], [10, 68, 7], [74, 44, 7], [21, 16, 7], [6, 45, 7], [46, 50, 7], [43, 44, 7], [36, 52, 7], [14, 23, 7], [66, 73, 7], [42, 29, 7], [2, 56, 7], [47, 26, 7], [47, 2, 7], [7, 38, 7], [13, 22, 7], [46, 43, 7], [66, 56, 7], [82, 75, 7]],
     [[0, 0, 7], [4, 4, 7], [0, 9, 7], [5, 7, 7], [3, 2, 7], [8, 4, 7], [5, 8, 7], [12, 12, 7], [3, 15, 7], [0, 4, 7], [2, 6, 7], [5, 0, 7], [8, 0, 7], [11, 18, 7], [16, 14, 7], [3, 3, 7], [0, 4, 7], [7, 13, 7], [24, 0, 7], [4, 21, 7], [1, 5, 7], [27, 26, 7], [10, 5, 7], [16, 9, 7], [19, 3, 7], [28, 16, 7], [1, 26, 7], [33, 12, 7], [27, 22, 7], [32, 35, 7], [10, 26, 7], [34, 30, 7], [28, 39, 7], [35, 38, 7], [31, 10, 7], [22, 16, 7], [27, 18, 7], [29, 37, 7], [19, 2, 7], [24, 0, 7], [19, 30, 7], [11, 23, 7], [42, 18, 7], [18, 2, 7], [14, 50, 7], [43, 45, 7], [6, 18, 7], [25, 34, 7], [55, 32, 7], [1, 54, 7], [40, 39, 7], [20, 11, 7], [43, 4, 7], [40, 58, 7], [32, 25, 7], [30, 23, 7], [40, 43, 7], [5, 64, 7], [14, 31, 7], [66, 36, 7], [25, 57, 7], [0, 68, 7], [27, 44, 7], [33, 70, 7], [8, 2, 7], [24, 21, 7], [42, 66, 7], [69, 49, 7], [11, 24, 7], [29, 15, 7], [4, 26, 7], [49, 38, 7], [3, 14, 7], [70, 28, 7], [32, 59, 7], [72, 72, 7], [21, 67, 7], [40, 31, 7], [34, 57, 7], [7, 38, 7], [13, 26, 7], [45, 40, 7], [57, 51, 7], [46, 75, 7]],
     [[6, 6, 7], [4, 5, 7], [6, 0, 7], [4, 0, 7], [9, 11, 7], [5, 9, 7], [1, 7, 7], [0, 11, 7], [11, 11, 7], [2, 11, 7], [1, 11, 7], [4, 6, 7], [19, 18, 7], [4, 11, 7], [21, 14, 7], [1, 19, 7], [6, 19, 7], [12, 5, 7], [5, 23, 7], [3, 12, 7], [9, 22, 7], [16, 28, 7], [19, 5, 7], [13, 28, 7], [30, 29, 7], [6, 17, 7], [20, 5, 7], [31, 0, 7], [18, 5, 7], [22, 3, 7], [19, 21, 7], [28, 16, 7], [21, 36, 7], [14, 17, 7], [18, 39, 7], [12, 34, 7], [30, 21, 7], [39, 40, 7], [32, 29, 7], [42, 34, 7], [41, 6, 7], [17, 33, 7], [48, 49, 7], [26, 32, 7], [14, 24, 7], [1, 12, 7], [30, 49, 7], [50, 22, 7], [50, 9, 7], [14, 5, 7], [57, 13, 7], [42, 32, 7], [0, 41, 7], [1, 37, 7], [10, 43, 7], [45, 34, 7], [1, 57, 7], [9, 34, 7], [61, 54, 7], [48, 11, 7], [7, 48, 7], [53, 33, 7], [52, 33, 7], [50, 3, 7], [70, 45, 7], [70, 38, 7], [27, 27, 7], [26, 48, 7], [68, 57, 7], [21, 76, 7], [70, 3, 7], [73, 36, 7], [25, 4, 7], [44, 55, 7], [22, 32, 7], [18, 53, 7], [38, 42, 7], [14, 81, 7], [46, 12, 7], [7, 38, 7], [21, 15, 7], [47, 36, 7], [51, 50, 7], [77, 78, 7]],
     [[4, 4, 7], [4, 8, 7], [0, 4, 7], [1, 6, 7], [8, 8, 7], [12, 10, 7], [12, 7, 7], [0, 0, 7], [14, 7, 7], [2, 12, 7], [7, 13, 7], [8, 14, 7], [8, 0, 7], [0, 7, 7], [16, 5, 7], [11, 17, 7], [21, 2, 7], [5, 16, 7], [25, 24, 7], [1, 16, 7], [5, 27, 7], [21, 4, 7], [28, 5, 7], [4, 0, 7], [9, 7, 7], [30, 1, 7], [1, 0, 7], [8, 34, 7], [19, 7, 7], [25, 34, 7], [33, 20, 7], [8, 29, 7], [38, 28, 7], [33, 13, 7], [37, 30, 7], [8, 34, 7], [6, 2, 7], [43, 28, 7], [0, 2, 7], [14, 3, 7], [41, 18, 7], [8, 34, 7], [9, 44, 7], [26, 35, 7], [50, 23, 7], [10, 28, 7], [52, 23, 7], [40, 29, 7], [0, 40, 7], [43, 38, 7], [9, 22, 7], [4, 35, 7], [32, 43, 7], [43, 46, 7], [40, 59, 7], [30, 61, 7], [52, 31, 7], [27, 42, 7], [64, 50, 7], [35, 42, 7], [26, 17, 7], [68, 67, 7], [46, 53, 7], [26, 40, 7], [70, 65, 7], [36, 15, 7], [53, 29, 7], [73, 71, 7], [75, 13, 7], [63, 75, 7], [75, 69, 7], [64, 77, 7], [37, 37, 7], [63, 12, 7], [67, 37, 7], [22, 29, 7], [65, 26, 7], [61, 76, 7], [25, 27, 7], [7, 38, 7], [20, 13, 7], [40, 42, 7], [54, 57, 7], [85, 78, 7]],
     [[3, 2, 7], [1, 4, 7], [0, 4, 7], [5, 5, 7], [5, 11, 7], [1, 1, 7], [0, 7, 7], [10, 0, 7], [2, 10, 7], [3, 6, 7], [4, 12, 7], [0, 13, 7], [4, 12, 7], [3, 5, 7], [4, 6, 7], [4, 10, 7], [10, 20, 7], [9, 10, 7], [25, 0, 7], [3, 25, 7], [9, 25, 7], [5, 13, 7], [8, 22, 7], [29, 10, 7], [4, 31, 7], [28, 13, 7], [31, 28, 7], [12, 9, 7], [11, 33, 7], [17, 35, 7], [2, 27, 7], [19, 37, 7], [19, 8, 7], [30, 0, 7], [41, 7, 7], [2, 12, 7], [11, 13, 7], [20, 21, 7], [32, 3, 7], [42, 1, 7], [33, 36, 7], [15, 48, 7], [23, 35, 7], [6, 50, 7], [51, 48, 7], [25, 21, 7], [32, 26, 7], [14, 20, 7], [49, 6, 7], [23, 14, 7], [52, 56, 7], [5, 31, 7], [58, 2, 7], [48, 43, 7], [29, 58, 7], [56, 43, 7], [63, 61, 7], [2, 14, 7], [64, 52, 7], [61, 14, 7], [30, 53, 7], [14, 32, 7], [37, 1, 7], [42, 48, 7], [71, 30, 7], [68, 13, 7], [41, 51, 7], [6, 19, 7], [41, 56, 7], [64, 49, 7], [46, 62, 7], [27, 61, 7], [5, 51, 7], [50, 45, 7], [29, 57, 7], [61, 18, 7], [73, 72, 7], [21, 39, 7], [85, 37, 7], [47, 39, 7], [15, 23, 7], [34, 36, 7], [66, 60, 7], [77, 84, 7]],
     [[1, 3, 7], [3, 3, 7], [5, 8, 7], [5, 1, 7], [3, 0, 7], [0, 5, 7], [4, 0, 7], [13, 9, 7], [14, 4, 7], [0, 14, 7], [11, 17, 7], [7, 14, 7], [1, 14, 7], [17, 16, 7], [18, 2, 7], [10, 2, 7], [21, 21, 7], [21, 8, 7], [7, 4, 7], [12, 25, 7], [13, 1, 7], [14, 19, 7], [28, 11, 7], [1, 15, 7], [10, 20, 7], [14, 28, 7], [22, 6, 7], [28, 33, 7], [3, 12, 7], [36, 10, 7], [3, 1, 7], [15, 26, 7], [36, 10, 7], [25, 1, 7], [3, 1, 7], [4, 41, 7], [39, 28, 7], [11, 32, 7], [9, 2, 7], [19, 37, 7], [36, 5, 7], [9, 20, 7], [10, 35, 7], [25, 21, 7], [2, 43, 7], [48, 2, 7], [52, 11, 7], [53, 6, 7], [42, 2, 7], [20, 2, 7], [31, 14, 7], [8, 46, 7], [27, 42, 7], [47, 5, 7], [22, 54, 7], [8, 37, 7], [54, 34, 7], [27, 49, 7], [54, 46, 7], [35, 42, 7], [24, 56, 7], [33, 68, 7], [6, 42, 7], [64, 33, 7], [16, 24, 7], [9, 69, 7], [67, 26, 7], [48, 60, 7], [38, 44, 7], [3, 59, 7], [36, 54, 7], [0, 38, 7], [69, 48, 7], [53, 80, 7], [19, 59, 7], [30, 16, 7], [10, 36, 7], [40, 31, 7], [34, 57, 7], [7, 38, 7], [24, 25, 7], [45, 35, 7], [62, 55, 7], [73, 47, 7]],
     [[5, 5, 7], [5, 1, 7], [3, 5, 7], [5, 2, 7], [4, 4, 7], [2, 9, 7], [7, 0, 7], [10, 8, 7], [14, 12, 7], [3, 12, 7], [12, 11, 7], [0, 12, 7], [16, 0, 7], [15, 12, 7], [4, 11, 7], [4, 20, 7], [0, 1, 7], [24, 21, 7], [12, 7, 7], [16, 13, 7], [18, 7, 7], [25, 3, 7], [22, 2, 7], [12, 20, 7], [1, 23, 7], [1, 18, 7], [33, 25, 7], [22, 29, 7], [0, 27, 7], [32, 25, 7], [26, 32, 7], [11, 22, 7], [3, 14, 7], [1, 14, 7], [26, 12, 7], [15, 22, 7], [1, 20, 7], [44, 4, 7], [20, 15, 7], [35, 17, 7], [13, 20, 7], [48, 11, 7], [13, 9, 7], [49, 10, 7], [6, 14, 7], [46, 35, 7], [53, 45, 7], [36, 42, 7], [49, 16, 7], [51, 52, 7], [7, 24, 7], [5, 16, 7], [53, 54, 7], [60, 50, 7], [6, 61, 7], [15, 15, 7], [56, 50, 7], [49, 26, 7], [8, 17, 7], [35, 42, 7], [60, 29, 7], [39, 31, 7], [68, 50, 7], [6, 62, 7], [40, 71, 7], [24, 0, 7], [39, 70, 7], [10, 39, 7], [28, 0, 7], [73, 7, 7], [55, 42, 7], [70, 72, 7], [6, 3, 7], [34, 56, 7], [22, 43, 7], [56, 71, 7], [75, 31, 7], [47, 18, 7], [51, 50, 7], [15, 62, 7], [19, 23, 7], [46, 41, 7], [62, 57, 7], [77, 79, 7]],
     [[5, 0, 7], [6, 5, 7], [8, 5, 7], [10, 0, 7], [4, 11, 7], [9, 2, 7], [9, 5, 7], [4, 0, 7], [6, 6, 7], [15, 2, 7], [3, 15, 7], [14, 10, 7], [0, 5, 7], [0, 14, 7], [2, 7, 7], [8, 3, 7], [15, 0, 7], [11, 3, 7], [16, 2, 7], [13, 9, 7], [4, 11, 7], [3, 25, 7], [13, 1, 7], [18, 30, 7], [24, 1, 7], [5, 24, 7], [32, 24, 7], [34, 10, 7], [4, 19, 7], [9, 14, 7], [18, 17, 7], [35, 24, 7], [1, 33, 7], [19, 37, 7], [8, 25, 7], [27, 33, 7], [39, 9, 7], [33, 44, 7], [14, 28, 7], [45, 32, 7], [16, 33, 7], [47, 0, 7], [16, 40, 7], [26, 44, 7], [51, 40, 7], [3, 6, 7], [2, 52, 7], [6, 29, 7], [46, 41, 7], [54, 12, 7], [48, 45, 7], [54, 27, 7], [57, 30, 7], [32, 13, 7], [60, 58, 7], [3, 48, 7], [5, 60, 7], [34, 9, 7], [43, 54, 7], [12, 17, 7], [67, 17, 7], [24, 18, 7], [6, 33, 7], [68, 55, 7], [3, 1, 7], [2, 69, 7], [23, 73, 7], [50, 41, 7], [59, 16, 7], [51, 65, 7], [71, 74, 7], [55, 64, 7], [20, 21, 7], [25, 51, 7], [58, 25, 7], [34, 44, 7], [7, 79, 7], [9, 15, 7], [71, 43, 7], [77, 48, 7], [12, 24, 7], [47, 40, 7], [59, 62, 7], [78, 84, 7]],
     [[7, 4, 7], [1, 6, 7], [5, 9, 7], [4, 6, 7], [9, 2, 7], [0, 7, 7], [8, 5, 7], [4, 0, 7], [10, 5, 7], [8, 2, 7], [17, 3, 7], [0, 18, 7], [0, 15, 7], [7, 9, 7], [15, 4, 7], [5, 22, 7], [8, 20, 7], [19, 5, 7], [7, 1, 7], [3, 19, 7], [24, 5, 7], [8, 15, 7], [9, 10, 7], [16, 0, 7], [26, 8, 7], [8, 30, 7], [3, 7, 7], [1, 20, 7], [34, 26, 7], [10, 35, 7], [3, 27, 7], [17, 10, 7], [30, 19, 7], [11, 23, 7], [15, 21, 7], [33, 38, 7], [1, 5, 7], [12, 16, 7], [37, 44, 7], [32, 12, 7], [45, 34, 7], [15, 1, 7], [30, 45, 7], [0, 9, 7], [12, 25, 7], [49, 11, 7], [53, 52, 7], [6, 16, 7], [22, 2, 7], [17, 51, 7], [24, 57, 7], [38, 41, 7], [28, 18, 7], [54, 57, 7], [59, 15, 7], [4, 41, 7], [55, 6, 7], [40, 5, 7], [54, 36, 7], [8, 4, 7], [54, 62, 7], [59, 13, 7], [36, 20, 7], [27, 69, 7], [59, 8, 7], [56, 9, 7], [73, 66, 7], [0, 26, 7], [23, 72, 7], [71, 70, 7], [77, 56, 7], [0, 42, 7], [59, 78, 7], [30, 30, 7], [46, 18, 7], [38, 78, 7], [64, 34, 7], [33, 9, 7], [2, 54, 7], [29, 68, 7], [20, 22, 7], [28, 46, 7], [66, 61, 7], [87, 83, 7]]]

expected = [(np.array([0.0, 0.0, 0.08333333, 0.14583333]), np.array([0.0, 0.17307692, 0.44230769, 0.52884615])),
            (np.array([0.0, 0.47916667, 0.39583333, 0.54166667]), np.array([0.0, 0.48557692, 0.46634615, 0.57211538])),
            (np.array([0.0, 0.5, 0.5, 0.3125]), np.array([0.0, 0.41826923, 0.34615385, 0.51442308])),
            (np.array([0.0, 0.04166667, 0.10416667, 0.0625]), np.array([0.0, 0.10576923, 0.28365385, 0.52403846])),
            (np.array([0.0, 0.5, 0.58333333, 0.54166667]), np.array([0.0, 0.42307692, 0.51923077, 0.49038462])),
            (np.array([0.0, 0.08333333, 0.02083333, 0.125]), np.array([0.0, 0.05769231, 0.09134615, 0.57211538])),
            (np.array([0.0, 0.04166667, 0.125, 0.1875]), np.array([0.0, 0.06730769, 0.38461538, 0.50480769])),
            (np.array([0.0, 0.10416667, 0.08333333, 0.08333333]), np.array([0.0, 0.39903846, 0.35096154, 0.43269231])),
            (np.array([0.0, 0.41666667, 0.58333333, 0.54166667]), np.array([0.0, 0.40384615, 0.51923077, 0.50480769])),
            (np.array([0.0, 0.41666667, 0.41666667, 0.375]), np.array([0.0, 0.36538462, 0.50961538,  0.50961538]))]


def test_temp():
    f = function_from_name('per_output')

    for i in range(10):
        trg_state = chained_from_operator(G[i], ind[i], 4, 4, ADD, 1)
        test_state = chained_from_operator(G[i], ind[i], 4, 4, ADD, 1, exclude=True)
        trg_state.add_function(f)
        test_state.add_function(f)
        actual = (trg_state.function_value(f), test_state.function_value(f))
        np.testing.assert_array_almost_equal(expected[i][0], actual[0])
        np.testing.assert_array_almost_equal(expected[i][1], actual[1])

import os.path
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
import bitpacking.packing as pk
import boolnet.bintools.operator_iterator as opit
import boolnet.bintools.example_generator as exgen


class TestExampleGenerator:
    cache = dict()

    @fixture(params=['add4.npz', 'add8.npz', 'sub4.npz', 'sub8.npz',
                     'mul2.npz', 'mul3.npz', 'mul4.npz', 'mul6.npz',
                     'mul2f.npz', 'mul3f.npz', 'mul4f.npz', 'mul6f.npz'])
    def file_func_inst(self, request, test_location):
        return test_location, request.param

    @fixture(params=[True, False])
    def exclude(self, request):
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
        return name[:3], inp, tgt

    def build_generator_instance(self, instance, exclude):
        op, inp, tgt = self.load_file_func_instance(instance)

        Ne, Ni = inp.shape
        _, No = tgt.shape

        indices = np.random.choice(Ne, min(100, Ne-1), replace=False)
        indices = np.sort(indices)
        inp = inp[indices, :]
        tgt = tgt[indices, :]
        inp_p = pk.packmat(inp)
        tgt_p = pk.packmat(tgt)
        if exclude:
            indices = [i for i in range(Ne) if i not in indices]

        it = opit.operator_example_iterator(op, Ni//2, indices, exclude)

        return (it, inp_p, tgt_p)

    def test_packed_generation(self, file_func_inst, exclude):
        example_iter, expected_X, expected_Y = self.build_generator_instance(
            file_func_inst, exclude)

        actual_X = np.zeros_like(expected_X)
        actual_Y = np.zeros_like(expected_Y)
        Ni, cols = expected_X.shape
        No, _ = expected_Y.shape

        exgen.pack_examples(example_iter, actual_X, actual_Y)

        assert_array_equal(expected_X, actual_X)
        assert_array_equal(expected_Y, actual_Y)

    def test_packed_from_operator_inc(self, file_func_inst, exclude):
        op, inp, tgt = self.load_file_func_instance(file_func_inst)

        Ne, Ni = inp.shape
        _, No = tgt.shape

        indices = np.random.choice(Ne, min(100, Ne-1), replace=False)
        indices = np.sort(indices)
        if exclude:
            expected_indices = [i for i in range(Ne) if i not in indices]
        else:
            expected_indices = indices
        expected = pk.packmat(np.hstack((
            inp[expected_indices, :],
            tgt[expected_indices, :])))

        actual = exgen.packed_from_operator(
            indices=indices,
            Nb=Ni//2,
            No=No,
            operator=op,
            exclude=exclude
        )

        assert_array_equal(expected, actual)

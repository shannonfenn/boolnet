import numpy as np
from BoolNet.packing import PACKED_SIZE, packed_type, pack_chunk


class ExampleGenerator:

    def __init__(self, Ni, No, Ne, start_index, index_functor, input_functor, target_functor):
        self.Ne = Ne
        self.Ni = Ni
        self.No = No
        self.start_idx = start_index

        self.idx_f = index_functor
        self.inp_f = input_functor
        self.tgt_f = target_functor

        self.inp_block = np.zeros((Ne, PACKED_SIZE), dtype=np.uint8)
        self.out_block = np.zeros((Ne, PACKED_SIZE), dtype=np.uint8)
        self.inp_block_packed = np.zeros((1, No), dtype=packed_type)
        self.out_block_packed = np.zeros((1, No), dtype=packed_type)
        self.reset()

    def reset(self):
        self.e = 0
        self.cur_idx = self.start_index

    def next_examples(self, inputs, target):
        if self.e >= self.Ne:
            raise IndexError('ExampleGenerator - past end of examples.')
        cols, No = inputs.shape
        remaining_cols = (self.Ne - self.e) // PACKED_SIZE
        for c in range(min(cols, remaining_cols)):
            self._get_block(inputs, target, c)

    def _get_block(self, inputs, target, col):
        remaining = self.Ne - self.e
        for i in range(min(remaining, PACKED_SIZE)):
            self.inp_block[i, :] = self.inp_f(self.cur_idx)
            self.tgt_block[i, :] = self.tgt_f(self.cur_idx)
            self.cur_idx = self.idx_f(self.cur_idx)
            self.e += 1

        pack_chunk(self.inp_block, inputs, col)
        pack_chunk(self.tgt_block, target, col)

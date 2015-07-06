import numpy as np
from boolnet.bintools.packing import packed_type, pack_chunk
from boolnet.bintools.packing import PACKED_SIZE_PY as PACKED_SIZE


# cdef class PackedExampleGenerator:
#     ''' presently feature sizes greater than 64 are not handled.'''

#     # Ni_p = Ni // 64 if Ni % 64 == 0 else Ni // 64 + 1
#     # No_p = No // 64 if No % 64 == 0 else No // 64 + 1
#     # self.inp_block = np.zeros((PACKED_SIZE, Ni_p), dtype=np.uint64)
#     # self.out_block = np.zeros((PACKED_SIZE, No_p), dtype=np.uint64)
#     def __init__(self, No, example_factory):
#         self.No = No
#         self.Ne = example_factory.Ne
#         self.Ni = example_factory.Ni
#         self.example_factory = example_factory

#         self.inp_block = np.zeros(PACKED_SIZE, dtype=np.uint64)
#         self.out_block = np.zeros(PACKED_SIZE, dtype=np.uint64)
#         self.inp_block_packed = np.zeros(self.Ni, dtype=packed_type)
#         self.out_block_packed = np.zeros(self.No, dtype=packed_type)
#         self.reset()

#     cpdef reset(self):
#         self.example_iter = iter(self.example_factory)

#     cpdef next_examples(self, inputs, target):
#         remaining = len(self.example_iter)
#         if remaining == 0:
#             raise IndexError('ExampleGenerator - past end of examples.')
#         cols, No = inputs.shape
#         remaining_cols = remaining // PACKED_SIZE
#         for c in range(min(cols, remaining_cols)):
#             self._get_block(inputs, target, c)

#     cpdef _get_block(self, inputs, target, col):
#         remaining = len(self.example_iter)
#         for i in range(min(remaining, PACKED_SIZE)):
#             self.inp_block[i], self.tgt_block[i] = next(self.example_iter)

#         pack_chunk(self.inp_block, inputs, col)
#         pack_chunk(self.tgt_block, target, col)

#     cpdef __check_invariants(self):
#         if self.Ni > 64 or self.No > 64:
#             raise ValueError('Ni or No greater than 64.')
#         if not isinstance(self.No, int) or self.No <= 0:
#             raise ValueError('Invalid output width (must be a positive integer).')


# cdef class OperatorExampleFactory():
#     def __init__(self, generator_factory, operator, Ne, Nb, include):
#         self.gen_fac = generator_factory
#         self.op = operator
#         self.Ne = Ne
#         self.Nb = Nb
#         self.Ni = 2*Nb
#         self.include = include

#     cpdef __iter__(self):
#         if self.include:
#             return OperatorIncludeIterator(self.gen_fac(), self.op, self.Nb, self.Ne)
#         else:
#             return OperatorExcludeIterator(self.gen_fac(), self.op, self.Nb, self.Ne)

#     cpdef __len__(self):
#         return self.Ne

#     cpdef __check_invariants(self):
#         if not isinstance(self.Nb, int) or self.Nb <= 0:
#             raise ValueError('Invalid operand width (must be a positive integer).')
cdef class BinaryOperatorIterator:
    def __init__(self, Nb, num_elements):
        self.Nb = Nb
        self.divisor = 2**Nb
        self.remaining = num_elements

cdef class BinaryOperatorIncludeIterator(BinaryOperatorIterator):
    def __init__(self, include_iterator, Nb, num_elements):
        super().__init__(Nb, num_elements)
        # self.index_iter = include_iterator

    def __iter__(self):
        return self

    def __len__(self):
        return self.remaining


cdef class BinaryOperatorExcludeIterator(BinaryOperatorIterator):
    def __init__(self, exclude_iterator, Nb, num_elements):
        # self.exclude_iter = exclude_iterator
        self.index = 0

        # try:
        #     self.next_exclude = next(self.exclude_iter)
        # except StopIteration:
        #     self.next_exclude = -1

        # self._sync()

    def __iter__(self):
        return self


    cdef void _sync(self):
        if self.next_exclude >= 0:
            try:
                while self.index == self.next_exclude:
                    self.index += 1
                    self.next_exclude = next(self.exclude_iter)
            except StopIteration:
                # no more exlude indices
                self.next_exclude = -1

    def __len__(self):
        return self.remaining



cdef class AddIncludeOperator(BinaryOperatorIncludeIterator):
    def __next__(self):
        inp = next(self.index_iter)
        out = inp // self.divisor + inp % self.divisor
        self.remaining -= 1
        return inp, out

        
cdef class AddExcludeOperator(BinaryOperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        inp = self.index
        out = inp // self.divisor + inp % self.divisor
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out
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


cdef class OperatorExampleFactory:
    def __init__(self, size_t[:] indices, size_t Ne, size_t Nb,
                 Operator operator, bint inc):
        print('lah')
        self.indices = np.array(indices)
        self.op = operator
        self.Ne = Ne
        self.Nb = Nb
        self.Ni = 2*Nb
        self.inc = inc

    def __iter__(self):
        print('lah2')
        if self.inc:
            print('lah2True')
            if self.op == ADD:
                print('lah2Add')
                return AddIncludeIterator(self.indices, self.Nb)
            elif self.op == SUB:
                print('lah2Sub')
                return SubIncludeIterator(self.indices, self.Nb)
            if self.op == MUL:
                print('lah2Mul')
                return MulIncludeIterator(self.indices, self.Nb)
        else:
            print('lah2False')
            if self.op == ADD:
                return AddExcludeIterator(self.indices, self.Nb, self.Ne)
            elif self.op == SUB:
                return SubExcludeIterator(self.indices, self.Nb, self.Ne)
            if self.op == MUL:
                return MulExcludeIterator(self.indices, self.Nb, self.Ne)

    def __len__(self):
        return self.Ne


cdef class BinaryOperatorIterator:
    def __init__(self, Nb, num_elements):
        self.Nb = Nb
        self.divisor = 2**Nb
        self.remaining = num_elements

    def __iter__(self):
        return self

    def __len__(self):
        return self.remaining


cdef class BinaryOperatorIncludeIterator(BinaryOperatorIterator):
    def __init__(self, size_t[:] include_list, size_t Nb):
        super().__init__(Nb, include_list.size)
        self.include_iter = iter(include_list)



cdef class AddIncludeIterator(BinaryOperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) + (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class SubIncludeIterator(BinaryOperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out, upper, lower
        inp = next(self.include_iter)
        upper = (inp // self.divisor)
        lower = (inp % self.divisor)
        if upper >= lower:
            out = upper - lower
        else:
            out = self.divisor - lower + upper
        self.remaining -= 1
        return inp, out


cdef class MulIncludeIterator(BinaryOperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) * (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class BinaryOperatorExcludeIterator(BinaryOperatorIterator):
    def __init__(self, size_t[:] exclude_list, size_t Nb, size_t num_elements):
        super().__init__(Nb, num_elements - exclude_list.size)
        self.num_elements = num_elements
        self.ex_iter = iter(exclude_list)
        self.index = 0
        try:
            self.ex_index = next(self.ex_iter)
            self._sync()
        except StopIteration:
            self.ex_index = self.num_elements

    cdef void _sync(self):
        if self.ex_index < self.num_elements:
            try:
                while self.index == self.ex_index:
                    self.index += 1
                    self.ex_index = next(self.ex_iter)
            except StopIteration:
                self.ex_index = self.num_elements

        
cdef class AddExcludeIterator(BinaryOperatorExcludeIterator):
    def __next__(self):
        print('in')
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) + (inp % self.divisor)
        print(inp, out)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class SubExcludeIterator(BinaryOperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) - (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class MulExcludeIterator(BinaryOperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) * (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out
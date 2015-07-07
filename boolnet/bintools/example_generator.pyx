import numpy as np
from boolnet.bintools.packing cimport packed_type_t, pack_chunk, PACKED_SIZE
from boolnet.bintools.packing import packed_type


cdef class PackedExampleGenerator:
    ''' presently feature sizes greater than 64 are not handled.'''
    # Ni_p = Ni // 64 if Ni % 64 == 0 else Ni // 64 + 1
    # No_p = No // 64 if No % 64 == 0 else No // 64 + 1
    # self.inp_block = np.zeros((PACKED_SIZE, Ni_p), dtype=np.uint64)
    # self.tgt_block = np.zeros((PACKED_SIZE, No_p), dtype=np.uint64)
    def __init__(self, OperatorExampleFactory example_factory, size_t No):
        self.No = No
        self.Ne = example_factory.Ne
        self.Ni = example_factory.Ni
        self.example_factory = example_factory

        self.inp_block = np.zeros(PACKED_SIZE, dtype=packed_type)
        self.tgt_block = np.zeros(PACKED_SIZE, dtype=packed_type)
        self.reset()

    cpdef reset(self):
        self.example_iter = iter(self.example_factory)

    cpdef next_examples(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target):
        cdef size_t i, remaining, remaining_blocks, blocks
        remaining = len(self.example_iter)
        
        if remaining == 0:
            raise IndexError('ExampleGenerator - past end of examples.')
        if inputs.shape[0] != self.Ni:
            raise IndexError('ExampleGenerator - inputs does not match Ni in shape.')
        if target.shape[0] != self.No:
            raise IndexError('ExampleGenerator - target does not match No in shape.')
        
        blocks = inputs.shape[1]
        remaining_blocks = remaining // PACKED_SIZE
        if remaining % PACKED_SIZE:
            remaining_blocks += 1

        for i in range(min(blocks, remaining_blocks)):
            self._get_block(inputs, target, i)

    def __bool__(self):
        return len(self.example_iter) > 0

    # cdef void _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t col):
    cdef _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t block):
        cdef size_t remaining = len(self.example_iter)

        if remaining >= PACKED_SIZE:
            remaining = PACKED_SIZE

        for i in range(remaining):
            self.inp_block[i], self.tgt_block[i] = next(self.example_iter)

        for i in range(remaining, PACKED_SIZE):
            self.inp_block[i], self.tgt_block[i] = 0, 0

        pack_chunk(self.inp_block, inputs, self.Ni, block)
        pack_chunk(self.tgt_block, target, self.No, block)

    cdef void __check_invariants(self):
        if self.Ni > 64 or self.No > 64:
            raise ValueError('Ni or No greater than 64 not supported.')


cdef class OperatorExampleFactory:
    def __init__(self, indices, size_t Nb, Operator operator, size_t max_elements=0):
        self.__check_operator(operator)
        self.indices = np.array(indices, dtype=np.uintp)
        self.op = operator
        self.max_elements = max_elements
        self.inc = (max_elements == 0)
        if self.inc:
            self.Ne = self.indices.size
        else:
            if max_elements < self.indices.size:
                raise ValueError('Exclude list larger than max_elements.')
            self.Ne = max_elements - self.indices.size
        self.Nb = Nb
        if operator in [UNARY_AND, UNARY_OR]:
            self.Ni = Nb
        else:
            self.Ni = 2*Nb

    def __iter__(self):
        if self.inc:
            if self.op == ZERO:
                return ZeroIncludeIterator(self.indices, self.Nb)
            elif self.op == UNARY_AND:
                return UnaryANDIncludeIterator(self.indices, self.Nb)
            elif self.op == UNARY_OR:
                return UnaryORIncludeIterator(self.indices, self.Nb)
            elif self.op == AND:
                return ANDIncludeIterator(self.indices, self.Nb)
            elif self.op == OR:
                return ORIncludeIterator(self.indices, self.Nb)
            elif self.op == ADD:
                return AddIncludeIterator(self.indices, self.Nb)
            elif self.op == SUB:
                return SubIncludeIterator(self.indices, self.Nb)
            elif self.op == MUL:
                return MulIncludeIterator(self.indices, self.Nb)
            elif self.op == AND:
                return UnaryANDIncludeIterator(self.indices, self.Nb)
            elif self.op == OR:
                return UnaryORIncludeIterator(self.indices, self.Nb)
        else:
            if self.op == ZERO:
                return ZeroExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == UNARY_AND:
                return UnaryANDExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == UNARY_OR:
                return UnaryORExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == AND:
                return ANDExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == OR:
                return ORExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == ADD:
                return AddExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == SUB:
                return SubExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == MUL:
                return MulExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == AND:
                return UnaryANDExcludeIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == OR:
                return UnaryORExcludeIterator(self.indices, self.Nb, self.max_elements)

    def __len__(self):
        return self.Ne

    cdef __check_operator(self, Operator op):
        if op not in [ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL]:
            raise ValueError('Invalid operator value ({})'.format(op))


cdef class OperatorIterator:
    def __init__(self, size_t Nb, size_t Ne):
        if Nb == 0:
            raise ValueError('Zero bitwidth not allowed.')
        self.divisor = 2**Nb
        self.remaining = Ne

    def __iter__(self):
        return self

    def __len__(self):
        return self.remaining


cdef class OperatorIncludeIterator(OperatorIterator):
    def __init__(self, size_t[:] include_list, size_t Nb):
        super().__init__(Nb, include_list.size)
        self.include_iter = iter(include_list)


cdef class ZeroIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 0
        self.remaining -= 1
        return inp, out


cdef class UnaryANDIncludeIterator(OperatorIncludeIterator):
    def __init__(self, size_t[:] include_list, size_t Nb):
        super().__init__(include_list, Nb)
        self.all_ones = 2**Nb - 1

    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 1 if inp == self.all_ones else 0
        self.remaining -= 1
        return inp, out


cdef class UnaryORIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 0 if inp == 0 else 1
        self.remaining -= 1
        return inp, out


cdef class ANDIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) & (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class ORIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) | (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class AddIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) + (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class SubIncludeIterator(OperatorIncludeIterator):
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


cdef class MulIncludeIterator(OperatorIncludeIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) * (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class OperatorExcludeIterator(OperatorIterator):
    def __init__(self, size_t[:] exclude_list, size_t Nb, size_t num_elements):
        if num_elements < exclude_list.size:
            raise ValueError('Exclude list larger than num_elements.')
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


cdef class ZeroExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp = self.index
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, 0


cdef class UnaryANDExcludeIterator(OperatorExcludeIterator):
    def __init__(self, size_t[:] exclude_list, size_t Nb, size_t num_elements):
        super().__init__(exclude_list, Nb, num_elements)
        self.all_ones = 2**Nb - 1

    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = 1 if inp == self.all_ones else 0
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class UnaryORExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = 0 if inp == 0 else 1
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class ANDExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) & (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class ORExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) | (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class AddExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) + (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class SubExcludeIterator(OperatorExcludeIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        upper = (inp // self.divisor)
        lower = (inp % self.divisor)
        if upper >= lower:
            out = upper - lower
        else:
            out = self.divisor - lower + upper
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class MulExcludeIterator(OperatorExcludeIterator):
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
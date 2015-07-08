# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

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
        cdef size_t inp, out, upper, lower
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
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Operator:
    ZERO,
    AND,
    OR,
    UNARY_AND,
    UNARY_OR,
    ADD,
    SUB,
    MUL,


cdef class OperatorIterator:
    cdef size_t divisor, remaining


cdef class OperatorIncludeIterator(OperatorIterator):
    cdef object include_iter


cdef class ZeroIncludeIterator(OperatorIncludeIterator):
    pass


cdef class UnaryANDIncludeIterator(OperatorIncludeIterator):
    cdef size_t all_ones


cdef class UnaryORIncludeIterator(OperatorIncludeIterator):
    pass


cdef class ANDIncludeIterator(OperatorIncludeIterator):
    pass


cdef class ORIncludeIterator(OperatorIncludeIterator):
    pass


cdef class AddIncludeIterator(OperatorIncludeIterator):
    pass


cdef class SubIncludeIterator(OperatorIncludeIterator):
    pass


cdef class MulIncludeIterator(OperatorIncludeIterator):
    pass


cdef class OperatorExcludeIterator(OperatorIterator):    
    cdef:
        size_t index, ex_index, num_elements
        object ex_iter

    cdef void _sync(self)


cdef class ZeroExcludeIterator(OperatorExcludeIterator):
    pass


cdef class UnaryANDExcludeIterator(OperatorExcludeIterator):
    cdef size_t all_ones


cdef class UnaryORExcludeIterator(OperatorExcludeIterator):
    pass


cdef class ANDExcludeIterator(OperatorExcludeIterator):
    pass


cdef class ORExcludeIterator(OperatorExcludeIterator):
    pass


cdef class AddExcludeIterator(OperatorExcludeIterator):
    pass


cdef class SubExcludeIterator(OperatorExcludeIterator):
    pass


cdef class MulExcludeIterator(OperatorExcludeIterator):
    pass

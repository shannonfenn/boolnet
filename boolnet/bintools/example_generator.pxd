import numpy as np
from boolnet.bintools.packing cimport packed_type_t, pack_chunk, PACKED_SIZE


cpdef enum Operator:
    ZERO,
    AND,
    OR,
    UNARY_AND,
    UNARY_OR,
    ADD,
    SUB,
    MUL,


cdef class PackedExampleGenerator:
    ''' presently feature sizes greater than 64 are not handled.'''
    cdef:
        readonly size_t No, Ne, Ni
        OperatorExampleFactory example_factory
        packed_type_t[:] inp_block, tgt_block
        object example_iter

    cpdef reset(self)

    cpdef next_examples(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target)

    # cdef void _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t col)
    cdef _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t col)

    cdef void __check_invariants(self)


cdef class OperatorExampleFactory:
    cdef:
        size_t Ne, Nb, Ni, max_elements
        size_t[:] indices
        bint inc
        Operator op
    cdef __check_operator(self, Operator op)


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

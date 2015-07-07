import numpy as np
from boolnet.bintools.packing cimport packed_type_t, pack_chunk, PACKED_SIZE


cpdef enum Operator:
    ZERO,
    AND,
    OR,
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

cdef class BinaryOperatorIterator:
    cdef size_t divisor, remaining


cdef class BinaryOperatorIncludeIterator(BinaryOperatorIterator):
    cdef object include_iter


cdef class ZeroIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class AndIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class OrIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class AddIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class SubIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class MulIncludeIterator(BinaryOperatorIncludeIterator):
    pass


cdef class BinaryOperatorExcludeIterator(BinaryOperatorIterator):    
    cdef:
        size_t index, ex_index, num_elements
        object ex_iter

    cdef void _sync(self)


cdef class ZeroExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class AndExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class OrExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class AddExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class SubExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class MulExcludeIterator(BinaryOperatorExcludeIterator):
    pass

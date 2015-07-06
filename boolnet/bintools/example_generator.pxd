import numpy as np
from boolnet.bintools.packing import packed_type, pack_chunk
from boolnet.bintools.packing import PACKED_SIZE_PY as PACKED_SIZE


# cdef class PackedExampleGenerator:
#     ''' presently feature sizes greater than 64 are not handled.'''
#     cdef:
#         size_t No, Ne, Ni
#         object example_factory
#         packed_type_t[:] inp_block, out_block, inp_block_packed, out_block_packed

#     cpdef reset(self)

#     cpdef next_examples(self, inputs, target)

#     cdef void _get_block(self, inputs, target, col)

#     cdef void __check_invariants(self)

cpdef enum Operator:
    ADD,
    SUB,
    MUL

cdef class OperatorExampleFactory:
    cdef:
        size_t Ne, Nb, Ni
        size_t[:] indices
        bint inc
        Operator op


cdef class BinaryOperatorIterator:
    cdef size_t Nb, divisor, remaining


cdef class BinaryOperatorIncludeIterator(BinaryOperatorIterator):
    cdef object include_iter


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


cdef class AddExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class SubExcludeIterator(BinaryOperatorExcludeIterator):
    pass


cdef class MulExcludeIterator(BinaryOperatorExcludeIterator):
    pass

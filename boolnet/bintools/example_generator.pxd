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


# cdef class OperatorExampleFactory():
#     cdef:
#         size_t Ne, Nb, Ni
#         size_t[:] indices
#         bint inc

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
    cdef size_t Nb, divisor, remaining


cdef class BinaryOperatorIncludeIterator(BinaryOperatorIterator):
    cdef size_t[:] include_list


cdef class BinaryOperatorExcludeIterator(BinaryOperatorIterator):    
    cdef size_t[:] exclude_list
    cdef void _sync(self)


# cdef class AddIncludeOperator(BinaryOperatorIncludeIterator)

        
# cdef class AddExcludeOperator(BinaryOperatorExcludeIterator)

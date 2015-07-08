# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from boolnet.bintools.packing cimport packed_type_t
from boolnet.bintools.operator_iterator cimport Operator


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


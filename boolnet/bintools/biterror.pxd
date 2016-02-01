# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import cython
cimport numpy as np
from boolnet.bintools.packing cimport packed_type_t


cdef class StandardEvaluator:
    cdef size_t Ne, No, cols, start, step
    cdef double divisor


cdef class StandardMCC(StandardEvaluator):
    cdef packed_type_t[:] true_positive, false_positive, false_negative
    cdef double[:] mcc

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardPerOutput(StandardEvaluator):
    cdef double[:] accumulator

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardAccuracy(StandardEvaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE1(StandardEvaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE2(StandardEvaluator):
    cdef double[:] weight_vector

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE3(StandardEvaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE4(StandardEvaluator):
    cdef size_t end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE5(StandardEvaluator):
    cdef size_t row_width, end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE6(StandardEvaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE7(StandardEvaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E)

# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import cython
from BoolNet.packing cimport packed_type_t


cdef class StandardEvaluator:
    cdef size_t No, cols, start, step
    cdef double divisor


cdef class StandardPerOutput:
    cdef:
        size_t No, start, step
        double divisor
        double[:] accumulator

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
    cdef size_t row_width

    cpdef double evaluate(self, packed_type_t[:, ::1] E)


cdef class StandardE7(StandardE6):
    cpdef double evaluate(self, packed_type_t[:, ::1] E)
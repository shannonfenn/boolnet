# cython: language_level=3
import cython
cimport numpy as np
from boolnet.bintools.packing cimport packed_type_t


cdef double matthews_correlation_coefficient(size_t FP, size_t TP, size_t FN, size_t TN)


cdef class Evaluator:
    cdef size_t Ne, No, cols, start, step
    cdef double divisor


cdef class PerOutputMCC(Evaluator):
    cdef packed_type_t[:] true_positive, false_positive, false_negative
    cdef double[:] mcc
    cdef packed_type_t end_mask

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class PerOutputMean(Evaluator):
    cdef double[:] accumulator

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class MeanMCC:
    cdef PerOutputMCC per_out_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class Accuracy(Evaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE1(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE2(Evaluator):
    cdef double[:] weight_vector

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE3(Evaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE4(Evaluator):
    cdef size_t end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE5(Evaluator):
    cdef size_t row_width, end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE6(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE7(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)

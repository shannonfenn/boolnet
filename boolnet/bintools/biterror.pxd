# cython: language_level=3
import cython
cimport numpy as np
from boolnet.bintools.packing cimport packed_type_t

cdef confusion(packed_type_t[:] errors, packed_type_t[:] target, size_t Ne,
               packed_type_t[:] TP_buf, packed_type_t[:] FP_buf, packed_type_t[:] FN_buf)

cdef double matthews_corr_coef(size_t TP, size_t TN, size_t FP, size_t FN)

cdef bint is_permutation(size_t[:] P, size_t No)


cdef class Evaluator:
    cdef size_t Ne, No, cols
    cdef size_t[:] order
    cdef double divisor


cdef class PerOutputMCC(Evaluator):
    cdef packed_type_t[:] tp_buffer, fp_buffer, fn_buffer
    cdef np.uint8_t[:] errors_exist
    cdef double[:] mcc
    cdef packed_type_t end_mask

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class PerOutputMean(Evaluator):
    cdef double[:] accumulator

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class MeanMCC(Evaluator):
    cdef PerOutputMCC per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class Accuracy(Evaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE1(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE2(Evaluator):
    cdef double[:] weight_vector

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE2MCC(StandardE2):
    cdef PerOutputMCC per_output_evaluator

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


cdef class StandardE6MCC(Evaluator):
    cdef PerOutputMCC per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class StandardE7(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)

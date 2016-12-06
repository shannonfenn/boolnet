import cython
from bitpacking.packing cimport packed_type_t
cimport numpy as np


cdef class ChainedEvaluator:
    cdef size_t Ne, No, cols
    cdef size_t[:] order
    cdef double divisor


cdef class ChainedPerOutputMCC(ChainedEvaluator):
    cdef:
        packed_type_t[:] tp_buffer, fp_buffer, fn_buffer
        size_t[:] TP, FP, FN
        np.uint8_t[:] errors_exist
        double[:] mcc

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedPerOutputMean(ChainedEvaluator):
    cdef:
        np.uint64_t[:] row_accumulator
        double[:] accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedCorrectness(ChainedEvaluator):
    cdef:
        packed_type_t[:] row_disjunction
        np.uint64_t accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedMeanMCC(ChainedEvaluator):
    cdef ChainedPerOutputMCC per_output_evaluator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE1(ChainedEvaluator):
    cdef:
        np.uint64_t[:] row_accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE2(ChainedE1):
    cdef double[:] weight_vector

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE2MCC(ChainedE2):
    cdef ChainedPerOutputMCC per_output_evaluator

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE3(ChainedE1):
    cdef packed_type_t[:] row_disjunction

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)


cdef class ChainedE4(ChainedE1):
    cdef size_t row_width, end_subtractor

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE5(ChainedEvaluator):
    cdef size_t row_width, err_rows, row_accumulator, end_subtractor

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE6(ChainedE5):
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)


cdef class ChainedE6MCC(ChainedEvaluator):
    cdef ChainedPerOutputMCC per_output_evaluator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class ChainedE7(ChainedEvaluator):
    cdef size_t err_rows

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)

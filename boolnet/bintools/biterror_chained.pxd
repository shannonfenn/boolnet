import cython
from boolnet.bintools.packing cimport packed_type_t
cimport numpy as np


cdef class ChainedEvaluator:
    cdef size_t No, cols, start, step
    cdef double divisor
    cdef np.uint8_t[:] mask


cdef class ChainedPerOutput(ChainedEvaluator):
    cdef:
        np.uint64_t[:] row_accumulator
        double[:] accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)
    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedAccuracy(ChainedEvaluator):
    cdef:
        packed_type_t[:] row_disjunction
        np.uint64_t accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedE1(ChainedEvaluator):
    cdef:
        np.uint64_t[:] row_accumulator

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedE2(ChainedE1):
    cdef double[:] weight_vector

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedE3(ChainedE1):
    cdef packed_type_t[:] row_disjunction

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)


cdef class ChainedE4(ChainedE1):
    cdef size_t row_width, end_subtractor

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedE5(ChainedEvaluator):
    cdef size_t Ne, row_width, err_rows, row_accumulator, end_subtractor

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)


cdef class ChainedE6(ChainedE5):
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)


cdef class ChainedE7(ChainedEvaluator):
    cdef size_t Ne, err_rows

    cpdef reset(self)
    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=*)
    cpdef double final_evaluation(self, packed_type_t[:, ::1] E)

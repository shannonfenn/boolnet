# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from boolnet.bintools.metric_names import (
    E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
    E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT, Metric)
import numpy as np
cimport numpy as np
import cython
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET
from boolnet.bintools.packing import packed_type


CHAINED_EVALUATORS = {
    E1:  (ChainedE1, False),
    E2L: (ChainedE2, False), E2M: (ChainedE2, True),
    E3L: (ChainedE3, False), E3M: (ChainedE3, True),
    E4L: (ChainedE4, False), E4M: (ChainedE4, True),
    E5L: (ChainedE5, False), E5M: (ChainedE5, True),
    E6L: (ChainedE6, False), E6M: (ChainedE6, True),
    E7L: (ChainedE7, False), E7M: (ChainedE7, True),
    ACCURACY: (ChainedAccuracy, False),
    PER_OUTPUT: (ChainedPerOutput, False)
}


cdef class ChainedEvaluator:
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.No = No
        self.divisor = No * Ne
        self.cols = cols
        if msb:
            self.start = No-1
            self.step = -1
        else:
            self.start = 0
            self.step = 1


cdef class ChainedPerOutput:
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.No = No
        self.divisor = Ne
        self.row_accumulator = np.zeros(self.No, dtype=np.uint64)
        self.accumulator = np.zeros(self.No, dtype=np.float64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i
        for i in range(self.No):
            self.row_accumulator[i] += popcount_vector(E[i, :])

    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i

        self.partial_evaluation(E)
        for i in range(self.No):
            self.accumulator[i] = self.row_accumulator[i] / self.divisor
        return self.accumulator


cdef class ChainedAccuracy(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.divisor = Ne
        self.accumulator = 0
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef reset(self):
        self.accumulator = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            r += self.step
        self.accumulator += popcount_vector(self.row_disjunction)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E)
        return 1.0 - self.accumulator / self.divisor


cdef class ChainedE1(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_accumulator = np.zeros(No, dtype=np.uint64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i
        for i in range(self.No):
            self.row_accumulator[i] += popcount_vector(E[i, :])

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E)

        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor

        return result


cdef class ChainedE2(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.divisor = Ne * (No + 1.0) * No / 2.0
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E)
            
        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor * self.weight_vector[i] 

        return result


cdef class ChainedE3(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            self.row_accumulator[r] += popcount_vector(self.row_disjunction)
            r += self.step


cdef class ChainedE4(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, row_sum
        
        r = self.start
        if self.row_accumulator[r] > 0:
            self.row_accumulator[r] += self.row_width - end_sub
        else:
            self.row_accumulator[r] += floodcount_vector(E[r, :], end_sub)

        r += self.step
        for i in range(self.No-1):
            if self.row_accumulator[r] > 0:
                self.row_accumulator[r] += self.row_width - end_sub
            else:
                self.row_accumulator[r] = max(self.row_accumulator[r - self.step],
                                              floodcount_vector(E[r, :], end_sub))
            r += self.step

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, self.end_subtractor)

        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor
        return result 


cdef class ChainedE5(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.Ne = Ne
        self.row_width = self.cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width


        self.end_subtractor = self.cols * PACKED_SIZE - Ne % (self.cols * PACKED_SIZE)
        self.reset()

    cpdef reset(self):
        row_accumulator = 0
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.No - self.err_rows):
            row_sum = floodcount_vector(E[r, :], end_sub)
            if row_sum > 0:
                self.row_accumulator = row_sum
                self.err_rows = self.No - i
                return
            r += self.step

        # no errors in earlier rows so just accumulate the same row
        # if already some error add row_width as if all future errs are high
        # else store whatever floodcount finds
        if self.err_rows > 0:
            self.row_accumulator += self.row_width - end_sub
        else:
            self.row_accumulator = floodcount_vector(E[r, :], end_sub)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E, self.end_subtractor)
        if self.err_rows > 0:
            return self.Ne / self.divisor * (self.err_rows - 1) + self.row_accumulator / self.divisor
        else:
            return 0.0


cdef class ChainedE6(ChainedE5):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.end_subtractor = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.No - self.err_rows):
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                # more important feature has error so overwrite accumulator
                self.row_accumulator = row_sum
                self.err_rows = self.No - i
                return
            r = self.start + self.step

        # no errors in earlier rows so just accumulate the same row
        self.row_accumulator += popcount_vector(E[r, :])


cdef class ChainedE7(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.Ne = Ne
        self.reset()

    cpdef reset(self):
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, size_t end_sub=0):
        cdef size_t i, r, c
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.err_rows, self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    self.err_rows = self.No - i
                    return
            r += self.step

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E)
        return self.Ne / self.divisor * self.err_rows 

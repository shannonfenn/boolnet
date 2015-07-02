from BoolNet.metric_names import (
    E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
    E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT, Metric)
import numpy as np
cimport numpy as np
import cython
from BoolNet.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from BoolNet.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET
from BoolNet.packing import packed_type


chained_evaluators = {
    E1: ChainedE1,
    E2L: ChainedE2, E2M: ChainedE2,
    E3L: ChainedE3, E3M: ChainedE3,
    E4L: ChainedE4, E4M: ChainedE4,
    E5L: ChainedE5, E5M: ChainedE5,
    E6L: ChainedE6, E6M: ChainedE6,
    E7L: ChainedE7, E7M: ChainedE7,
    ACCURACY: ChainedAccuracy,
    PER_OUTPUT: ChainedPerOutput
}


cdef class ChainedEvaluator:
    cdef size_t No, cols, start, step
    cdef double divisor

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
    cdef:
        size_t No, start, step
        double divisor
        np.uint64_t[:] row_accumulator
        double[:] accumulator

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.No = No
        self.divisor = No * Ne
        self.accumulator = np.zeros(self.No, dtype=np.float64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
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
    cdef:
        packed_type_t[:] row_disjunction
        np.uint64_t accumulator

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)
        self.accumulator = 0

    cpdef reset(self):
        self.accumulator = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
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
        return self.accumulator / self.divisor


cdef class ChainedE1(ChainedEvaluator):
    cdef:
        np.uint64_t[:] row_accumulator

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_accumulator = np.zeros(No, dtype=np.uint64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
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
    cdef double[:] weight_vector

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
    cdef packed_type_t[:] row_disjunction

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            self.row_accumulator[r] += popcount_vector(self.row_disjunction)
            r += self.step


cdef class ChainedE4(ChainedE1):
    cdef size_t row_width, end_subtractor

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
        cdef size_t i, r, row_sum
        
        r = self.start
        if self.row_accumulator[r] >= 0:
            self.row_accumulator[r] += self.row_width - <int>last * self.end_subtractor
        else:
            self.row_accumulator[r] += floodcount_vector(E[r, :], <int>last * self.end_subtractor)

        r += self.step
        for i in range(self.No-1):
            if self.row_accumulator[r] >= 0:
                self.row_accumulator[r] += self.row_width - <int>last * self.end_subtractor
            else:
                self.row_accumulator[r] = max(self.row_accumulator[r - 1],
                                              floodcount_vector(E[r, :], <int>last * self.end_subtractor))
            r += self.step

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, 1)

        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor
        return result


cdef class ChainedE5(ChainedEvaluator):
    cdef size_t row_width, err_rows, row_accumulator, end_subtractor

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = self.cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width
        self.reset()

    cpdef reset(self):
        row_accumulator = 0
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.err_rows, self.No):
            row_sum = floodcount_vector(E[r, :], <int>last * self.end_subtractor)
            if row_sum > 0:
                self.row_accumulator = row_sum
                self.err_rows = i
                return
            r += self.step

        # no errors in earlier rows so just accumulate the same row
        # if already some error add row_width as if all future errs are high
        # else store whatever floodcount finds
        if self.err_rows > 0:
            self.row_accumulator += self.row_width - <int>last * self.end_subtractor
        else:
            self.row_accumulator = floodcount_vector(E[r, :], <int>last * self.end_subtractor)

    cpdef last_calc(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E, 1)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        self.last_calc(E)
        if self.err_rows > 0:
            return (self.row_width - self.end_subtractor) / self.divisor * (self.err_rows - 1) + self.row_accumulator
        else:
            return 0.0


cdef class ChainedE6(ChainedE5):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.err_rows, self.No):
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                # more important feature has error so overwrite accumulator
                self.row_accumulator = row_sum
                self.err_rows = i
                return
            r = self.start + self.step

        # no errors in earlier rows so just accumulate the same row
        self.row_accumulator += popcount_vector(E[r, :])

    cpdef last_calc(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E)


cdef class ChainedE7(ChainedEvaluator):
    cdef size_t row_width, err_rows

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = self.cols * PACKED_SIZE
        self.row_width -= self.row_width - Ne % self.row_width
        self.reset()

    cpdef reset(self):
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, bint last=False):
        cdef size_t i, r, c
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.err_rows, self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    self.err_rows = i
                    return
            r += self.step

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E):
        self.partial_evaluation(E)
        return self.row_width / self.divisor * self.err_rows 

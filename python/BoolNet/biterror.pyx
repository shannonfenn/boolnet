
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from BoolNet.BitError import (E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
                              E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT, Metric)
import numpy as np
cimport numpy as np
import cython
from BoolNet.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from BoolNet.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET
from BoolNet.packing import packed_type


standard_evaluators = {
    E1:  StandardEvaluator,
    E2L: StandardE2, E2M: StandardE2,
    E3L: StandardE3, E3M: StandardE3,
    E4L: StandardE4, E4M: StandardE4,
    E5L: StandardE5, E5M: StandardE5,
    E6L: StandardE6, E6M: StandardE6,
    E7L: StandardE7, E7M: StandardE7,
    ACCURACY: StandardAccuracy,
    PER_OUTPUT: StandardPerOutput
}


cdef class StandardEvaluator:
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


cdef class StandardPerOutput:
    cdef:
        size_t No, start, step
        double divisor
        double[:] accumulator

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.No = No
        self.divisor = No * Ne
        self.accumulator = np.zeros(self.No, dtype=np.float64)

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i

        for i in range(self.No):
            self.accumulator[i] = popcount_vector(E[i, :]) / self.divisor
        return self.accumulator


cdef class StandardAccuracy(StandardEvaluator):
    cdef packed_type_t[:] row_disjunction

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            r += self.step

        return popcount_vector(self.row_disjunction) / self.divisor


cdef class StandardE1(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        for i in range(self.No):
            result += popcount_matrix(E) / self.divisor

        return result


cdef class StandardE2(StandardEvaluator):
    cdef double[:] weight_vector

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.divisor = Ne * (No + 1.0) * No / 2.0
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0
            
        for i in range(self.No):
            result += popcount_vector(E[i, :]) / self.divisor * self.weight_vector[i] 

        return result


cdef class StandardE3(StandardEvaluator):
    cdef packed_type_t[:] row_disjunction

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        cdef double result = 0.0
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            result += popcount_vector(self.row_disjunction) / self.divisor
            r += self.step
        return result


cdef class StandardE4(StandardEvaluator):
    cdef size_t end_subtractor

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        row_width = cols * PACKED_SIZE
        self.end_subtractor = row_width - Ne % row_width

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        cdef double result = 0.0
        
        r = self.start
        row_sum = floodcount_vector(E[r, :], self.end_subtractor)
        result = row_sum / self.divisor

        r += self.step
        for i in range(self.No-1):
            row_sum = max(row_sum, floodcount_vector(E[r, :], self.end_subtractor))
            result += row_sum / self.divisor
            r += self.step

        return result


cdef class StandardE5(StandardEvaluator):
    cdef size_t row_width, end_subtractor

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = self.cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width
        self.row_width -= self.end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = floodcount_vector(E[r, :], self.end_subtractor)
            if row_sum > 0:
                return (self.row_width / self.divisor * i) + (row_sum / self.divisor)
            r += self.step
        return 0.0


cdef class StandardE6(StandardEvaluator):
    cdef size_t row_width

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = self.cols * PACKED_SIZE
        self.row_width -= self.row_width - Ne % self.row_width

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                return (self.row_width / self.divisor * i) + (row_sum / self.divisor)
            r += self.step
        return 0.0


cdef class StandardE7(StandardE6):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    return self.row_width / self.divisor * (i + 1)
            r += self.step
        return 0.0

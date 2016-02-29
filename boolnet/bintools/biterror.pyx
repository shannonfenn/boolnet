# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from boolnet.bintools.functions import (E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
                                        E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT, MCC)
import numpy as np
cimport numpy as np
import cython
from math import sqrt
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET
from boolnet.bintools.packing import packed_type


STANDARD_EVALUATORS = {
    E1:  (StandardE1, False),
    E2L: (StandardE2, False), E2M: (StandardE2, True),
    E3L: (StandardE3, False), E3M: (StandardE3, True),
    E4L: (StandardE4, False), E4M: (StandardE4, True),
    E5L: (StandardE5, False), E5M: (StandardE5, True),
    E6L: (StandardE6, False), E6M: (StandardE6, True),
    E7L: (StandardE7, False), E7M: (StandardE7, True),
    ACCURACY: (StandardAccuracy, False),
    PER_OUTPUT: (StandardPerOutput, False),
    MCC: (StandardMCC, False)
}


cdef class StandardEvaluator:
    def __init__(self, size_t Ne, size_t No, bint msb):
        if Ne % PACKED_SIZE == 0:
            self.cols = Ne // PACKED_SIZE
        else:
            self.cols = Ne // PACKED_SIZE + 1
        if msb:
            self.start = No-1
            self.step = -1
        else:
            self.start = 0
            self.step = 1
        self.Ne = Ne
        self.No = No
        self.divisor = No * Ne


cdef class StandardMCC(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        self.true_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_negative = np.zeros(self.cols, dtype=packed_type)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        super().__init__(Ne, No, msb)

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] O, packed_type_t[:, ::1] T):
        cdef size_t i, c, TP, FP, TN, FN, normaliser
        
        TP = FP = TN = FN = 0

        for i in range(self.No):
            for c in range(self.cols):
                self.true_positive[c] = O[i, c] & T[i, c]
                self.false_positive[c] = O[i, c] & ~T[i, c]
                self.false_negative[c] = ~O[i, c] & T[i, c]
            TP = popcount_vector(self.true_positive)
            FP = popcount_vector(self.false_positive)
            FN = popcount_vector(self.false_negative)
            TN = self.Ne - TP - FP - FN
            normaliser = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            if normaliser == 0:
                self.mcc[i] = 0
            else:
                self.mcc[i] = (TP * TN - FP * FN) / sqrt(normaliser)
        return self.mcc


cdef class StandardPerOutput(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.accumulator = np.zeros(self.No, dtype=np.float64)
        self.divisor = Ne

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i

        for i in range(self.No):
            self.accumulator[i] = popcount_vector(E[i, :]) / self.divisor
        return self.accumulator


cdef class StandardAccuracy(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)
        self.divisor = Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            r += self.step

        return 1.0 - popcount_vector(self.row_disjunction) / self.divisor


cdef class StandardE1(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef double result = 0.0
        return popcount_matrix(E) / self.divisor


cdef class StandardE2(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)
        super().__init__(Ne, No, msb)
        self.divisor = Ne * (No + 1.0) * No / 2.0

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0
            
        for i in range(self.No):
            result += popcount_vector(E[i, :]) * self.weight_vector[i] 

        return result / self.divisor


cdef class StandardE3(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        cdef double result = 0.0
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            result += popcount_vector(self.row_disjunction)
            r += self.step
        return result / self.divisor


cdef class StandardE4(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        cdef double result
        
        r = self.start
        row_sum = floodcount_vector(E[r, :], self.end_subtractor)
        result = row_sum

        r += self.step
        for i in range(self.No-1):
            row_sum = max(row_sum, floodcount_vector(E[r, :], self.end_subtractor))
            result += row_sum
            r += self.step

        return result / self.divisor


cdef class StandardE5(StandardEvaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne % (self.cols * PACKED_SIZE)

    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = floodcount_vector(E[r, :], self.end_subtractor)
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
            r += self.step
        return 0.0


cdef class StandardE6(StandardEvaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
            r += self.step
        return 0.0


cdef class StandardE7(StandardEvaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E):
        cdef size_t i, r, c
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    return self.Ne / self.divisor * (self.No - i)
            r += self.step
        return 0.0

# cython: language_level=3
from boolnet.bintools.functions cimport (E1, E2M, E2L, E3M, E3L, E4M, E4L, E5M, E5L,
            E6M, E6L, E7M, E7L, ACCURACY, MCC, PER_OUTPUT_ERROR, PER_OUTPUT_MCC)
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET, generate_end_mask
from boolnet.bintools.packing import packed_type


EVALUATORS = {
    E1:  (StandardE1, False),
    E2L: (StandardE2, False), E2M: (StandardE2, True),
    E3L: (StandardE3, False), E3M: (StandardE3, True),
    E4L: (StandardE4, False), E4M: (StandardE4, True),
    E5L: (StandardE5, False), E5M: (StandardE5, True),
    E6L: (StandardE6, False), E6M: (StandardE6, True),
    E7L: (StandardE7, False), E7M: (StandardE7, True),
    ACCURACY: (Accuracy, False),
    MCC: (MeanMCC, False),
    PER_OUTPUT_ERROR: (PerOutputMean, False),
    PER_OUTPUT_MCC: (PerOutputMCC, False)
}


cdef double matthews_correlation_coefficient(size_t FP, size_t TP, size_t FN, size_t TN):
    cdef size_t actual_positives, actual_negatives, normaliser
    cdef double d

    actual_positives = (TP + FN)
    actual_negatives = (TN + FP)
    normaliser = actual_positives * actual_negatives * (TP + FP) * (TN + FN)
    if actual_positives == 0:
        # only one given class give accuracy in [-1, 1]
        return TN / <double>actual_negatives * 2 - 1
    elif actual_negatives == 0:
        # only one given class give accuracy in [-1, 1]
        return TP / <double>actual_positives * 2 - 1
    elif normaliser == 0:
        # normal limitting case when two classes present but only one predicted
        return 0
    else:
        # MCC = (TP * TN - FP * FN) / sqrt(normaliser)
        # below method has slight numerical inaccuracy but reduces overflow risk
        d = sqrt(sqrt(normaliser))
        return TP/d * TN/d - FP/d * FN/d



cdef class Evaluator:
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


cdef class PerOutputMCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.true_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_negative = np.zeros(self.cols, dtype=packed_type)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        self.end_mask = generate_end_mask(Ne)

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, c, FP, TP, FN
        cdef size_t actual_positives, actual_negatives, normaliser
        
        for i in range(self.No):
            for c in range(self.cols):
                self.false_positive[c] = E[i, c] & ~T[i, c]
                self.true_positive[c] = ~E[i, c] & T[i, c]
                self.false_negative[c] = E[i, c] & T[i, c]
            # ensure we do not count past Ne
            self.false_positive[-1] &= self.end_mask
            self.true_positive[-1] &= self.end_mask
            self.false_negative[-1] &= self.end_mask
                
            FP = popcount_vector(self.false_positive)
            TP = popcount_vector(self.true_positive)
            FN = popcount_vector(self.false_negative)
            
            self.mcc[i] = matthews_correlation_coefficient(
                FP, TP, FN, self.Ne - TP - FP - FN)
        return self.mcc


cdef class PerOutputMean(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.accumulator = np.zeros(self.No, dtype=np.float64)
        self.divisor = Ne

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i

        for i in range(self.No):
            self.accumulator[i] = popcount_vector(E[i, :]) / self.divisor
        return self.accumulator


cdef class MeanMCC:
    def __init__(self, size_t Ne, size_t No, bint msb):
        self.per_out_evaluator = PerOutputMCC(Ne, No, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        return sum(self.per_out_evaluator.evaluate(E, T)) / <double>self.per_out_evaluator.No


cdef class Accuracy(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)
        self.divisor = Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            r += self.step

        return 1.0 - popcount_vector(self.row_disjunction) / self.divisor


cdef class StandardE1(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef double result = 0.0
        return popcount_matrix(E) / self.divisor


cdef class StandardE2(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)
        super().__init__(Ne, No, msb)
        self.divisor = Ne * (No + 1.0) * No / 2.0

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
            
        for i in range(self.No):
            result += popcount_vector(E[i, :]) * self.weight_vector[i] 

        return result / self.divisor


cdef class StandardE3(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
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


cdef class StandardE4(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
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


cdef class StandardE5(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne % (self.cols * PACKED_SIZE)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = floodcount_vector(E[r, :], self.end_subtractor)
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
            r += self.step
        return 0.0


cdef class StandardE6(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
            r += self.step
        return 0.0


cdef class StandardE7(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, c
        
        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    return self.Ne / self.divisor * (self.No - i)
            r += self.step
        return 0.0

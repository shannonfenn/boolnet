# cython: language_level=3
# cython: boundscheck=False, nonecheck=False, cdivision=True, initializedcheck=False
from boolnet.bintools.functions cimport (E1, E2M, E2L, E3M, E3L, E4M, E4L, E5M, E5L,
            E6M, E6L, E7M, E7L, ACCURACY, MCC, PER_OUTPUT_ERROR, PER_OUTPUT_MCC)
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET, generate_end_mask
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
    MCC: (ChainedMeanMCC, False),
    PER_OUTPUT_ERROR: (ChainedPerOutputMean, False),
    PER_OUTPUT_MCC: (ChainedPerOutputMCC, False)
}



cdef class ChainedEvaluator:
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.Ne = Ne
        self.No = No
        self.divisor = No * Ne
        self.cols = cols
        if msb:
            self.start = No-1
            self.step = -1
        else:
            self.start = 0
            self.step = 1


cdef class ChainedPerOutputMCC(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.true_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_positive = np.zeros(self.cols, dtype=packed_type)
        self.false_negative = np.zeros(self.cols, dtype=packed_type)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        self.TP = np.zeros(self.No, dtype=packed_type)
        self.FP = np.zeros(self.No, dtype=packed_type)
        self.FN = np.zeros(self.No, dtype=packed_type)
        self.end_mask = generate_end_mask(Ne)

    cpdef reset(self):
        self.TP[...] = 0
        self.FP[...] = 0
        self.FN[...] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, packed_type_t end_mask=PACKED_ALL_SET):
        cdef size_t i, c

        for i in range(self.No):
            for c in range(self.cols):
                self.true_positive[c] = ~E[i, c] & T[i, c]
                self.false_positive[c] = E[i, c] & ~T[i, c]
                self.false_negative[c] = E[i, c] & T[i, c]
            # ensure we do not count past Ne
            self.true_positive[-1] &= end_mask
            self.false_positive[-1] &= end_mask
            self.false_negative[-1] &= end_mask

            self.TP[i] += popcount_vector(self.true_positive)
            self.FP[i] += popcount_vector(self.false_positive)
            self.FN[i] += popcount_vector(self.false_negative)

    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, TP, TN, FP, FN, normaliser

        self.partial_evaluation(E, T, self.end_mask)

        for i in range(self.No):
            TP, FP, FN = self.TP[i], self.FP[i], self.FN[i]
            TN = self.Ne - TP - FP - FN

            normaliser = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            if normaliser == 0:
                self.mcc[i] = 0
            else:
                self.mcc[i] = (TP * TN - FP * FN) / sqrt(normaliser)
        return self.mcc


cdef class ChainedPerOutputMean(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.No = No
        self.divisor = Ne
        self.row_accumulator = np.zeros(self.No, dtype=np.uint64)
        self.accumulator = np.zeros(self.No, dtype=np.float64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i
        for i in range(self.No):
            self.row_accumulator[i] += popcount_vector(E[i, :])

    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i

        self.partial_evaluation(E, T)
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

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        r = self.start
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            r += self.step
        self.accumulator += popcount_vector(self.row_disjunction)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.partial_evaluation(E, T)
        return 1.0 - self.accumulator / self.divisor


cdef class ChainedMeanMCC:
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.per_out_evaluator = ChainedPerOutputMCC(Ne, No, cols, msb)

    cpdef reset(self):
        self.per_out_evaluator.reset()

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.per_out_evaluator.partial_evaluation(E, T)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        return self.per_out_evaluator.final_evaluation(E, T).mean()


cdef class ChainedE1(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_accumulator = np.zeros(No, dtype=np.uint64)

    cpdef reset(self):
        self.row_accumulator[:] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i
        for i in range(self.No):
            self.row_accumulator[i] += popcount_vector(E[i, :])

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, T)

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

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, T)
            
        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor * self.weight_vector[i] 

        return result


cdef class ChainedE3(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
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

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
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

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, T, self.end_subtractor)

        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor
        return result 


cdef class ChainedE5(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.row_width = self.cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width
        self.reset()

    cpdef reset(self):
        row_accumulator = 0
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
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

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.partial_evaluation(E, T, self.end_subtractor)
        if self.err_rows > 0:
            return self.Ne / self.divisor * (self.err_rows - 1) + self.row_accumulator / self.divisor
        else:
            return 0.0


cdef class ChainedE6(ChainedE5):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.end_subtractor = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
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
            r += self.step

        # no errors in earlier rows so just accumulate the same row
        self.row_accumulator += popcount_vector(E[r, :])


cdef class ChainedE7(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        self.reset()

    cpdef reset(self):
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, c
        
        # first check if an earlier row now has an error value
        r = self.start
        for i in range(self.err_rows, self.No):
            for c in range(self.cols):
                if E[r, c] > 0:
                    self.err_rows = self.No - i
                    return
            r += self.step

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.partial_evaluation(E, T)
        return self.Ne / self.divisor * self.err_rows 

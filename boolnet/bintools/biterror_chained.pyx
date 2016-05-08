# cython: language_level=3
# cython: boundscheck=False, nonecheck=False, cdivision=True, initializedcheck=False
import boolnet.bintools.functions as fn
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt
from boolnet.bintools.biterror cimport confusion, matthews_corr_coef
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET, generate_end_mask
from boolnet.bintools.packing import packed_type


CHAINED_EVALUATORS = {
    fn.E1: ChainedE1,
    fn.E2: ChainedE2,
    fn.E3: ChainedE3,
    fn.E4: ChainedE4,
    fn.E5: ChainedE5,
    fn.E6: ChainedE6,
    fn.E7: ChainedE7,
    fn.E1_MCC: ChainedMeanMCC,
    fn.E2_MCC: ChainedE2MCC,
    fn.E6_MCC: ChainedE6MCC,
    fn.ACCURACY: ChainedAccuracy,
    fn.PER_OUTPUT_ERROR: ChainedPerOutputMean,
    fn.PER_OUTPUT_MCC: ChainedPerOutputMCC
}



cdef class ChainedEvaluator:
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        self.Ne = Ne
        self.No = No
        self.divisor = No * Ne
        self.cols = cols
        # check feature_order is a valid permutation
        assert len(feature_order) == No
        assert min(feature_order) == 0
        assert max(feature_order) == No - 1
        assert len(np.unique(feature_order)) == No
        self.order = np.array(feature_order)


cdef class ChainedAccuracy(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.divisor = Ne
        self.accumulator = 0
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef reset(self):
        self.accumulator = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, c
        
        self.row_disjunction[:] = 0
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[i, c]
        self.accumulator += popcount_vector(self.row_disjunction)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.partial_evaluation(E, T)
        return 1.0 - self.accumulator / self.divisor


cdef class ChainedPerOutputMCC(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.tp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fn_buffer = np.zeros(self.cols, dtype=packed_type)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        self.TP = np.zeros(self.No, dtype=np.uintp)
        self.FP = np.zeros(self.No, dtype=np.uintp)
        self.FN = np.zeros(self.No, dtype=np.uintp)
        self.errors_exist = np.zeros(self.No, dtype=np.uint8)

    cpdef reset(self):
        self.TP[...] = 0
        self.FP[...] = 0
        self.FN[...] = 0
        self.errors_exist[...] = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, TP, FP, FN

        for i in range(self.No):
            TP, _, FP, FN = confusion(
                E[i, :], T[i, :], self.Ne, self.tp_buffer,
                self.fp_buffer, self.fn_buffer)

            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN

    cpdef double[:] final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, FP, TP, FN, TN

        self.partial_evaluation(E, T)

        for i in range(self.No):
            TP, FP, FN = self.TP[i], self.FP[i], self.FN[i]

            self.errors_exist[i] = (FP + FN) > 0

            self.mcc[i] = matthews_corr_coef(
                TP, self.Ne - TP - FP - FN, FP, FN)

        return self.mcc


cdef class ChainedPerOutputMean(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
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


cdef class ChainedMeanMCC(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.per_output_evaluator = ChainedPerOutputMCC(Ne, No, cols, feature_order)

    cpdef reset(self):
        self.per_output_evaluator.reset()

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.per_output_evaluator.partial_evaluation(E, T)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        return sum(self.per_output_evaluator.final_evaluation(E, T)) / <double> self.No


cdef class ChainedE1(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
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
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        cdef size_t i
        super().__init__(Ne, No, cols, feature_order)
        # highest weight to earliest features - normalises to [0, 1]
        weights = np.arange(No, 0, -1, dtype=np.float64) / (0.5 * No * (No + 1.0))
        self.weight_vector = np.empty(No, dtype=np.float64)
        # reorder by feature order
        for i in range(No):
            self.weight_vector[i] = weights[self.order[i]]

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, T)            
        for i in range(self.No):
            result += self.row_accumulator[i] * self.weight_vector[i] / self.Ne
        return result


cdef class ChainedE2MCC(ChainedE2):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.per_output_evaluator = ChainedPerOutputMCC(Ne, No, cols, feature_order)

    cpdef reset(self):
        self.per_output_evaluator.reset()

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        self.per_output_evaluator.partial_evaluation(E, T)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
        cdef double[:] per_output

        per_output = self.per_output_evaluator.final_evaluation(E, T)

        for i in range(self.No):
            result += per_output[i] * self.weight_vector[i]
        return result


cdef class ChainedE3(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.row_disjunction = np.zeros(cols, dtype=packed_type)

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, c
        
        self.row_disjunction[:] = 0
        for i in range(self.No):
            r = self.order[i]
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            self.row_accumulator[r] += popcount_vector(self.row_disjunction)


cdef class ChainedE4(ChainedE1):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.row_width = cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, row_sum = 0
        
        for i in range(self.No):
            r = self.order[i]
            if self.row_accumulator[r] > 0:
                self.row_accumulator[r] += self.row_width - end_sub
            else:
                row_sum = max(row_sum,
                              floodcount_vector(E[r, :], end_sub))
                self.row_accumulator[r] = row_sum

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E, T, self.end_subtractor)

        for i in range(self.No):
            result += self.row_accumulator[i] / self.divisor
        return result 


cdef class ChainedE5(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.row_width = self.cols * PACKED_SIZE
        self.end_subtractor = self.row_width - Ne % self.row_width
        self.reset()

    cpdef reset(self):
        row_accumulator = 0
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        for i in range(self.No - self.err_rows):
            r = self.order[i]
            row_sum = floodcount_vector(E[r, :], end_sub)
            if row_sum > 0:
                self.row_accumulator = row_sum
                self.err_rows = self.No - i
                return

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
            return (self.err_rows - 1) / <double>self.No + self.row_accumulator / <double>self.divisor
            #return (self.err_rows - 1) / self.divisor * self.Ne + self.row_accumulator / self.divisor
            #return self.Ne / self.divisor * (self.err_rows - 1) + self.row_accumulator / self.divisor
        else:
            return 0.0


cdef class ChainedE6(ChainedE5):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.end_subtractor = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, row_sum
        
        # first check if an earlier row now has an error value
        for i in range(self.No - self.err_rows):
            r = self.order[i]
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                # more important feature has error so overwrite accumulator
                self.row_accumulator = row_sum
                self.err_rows = self.No - i
                return

        # no errors in earlier rows so just accumulate the same row
        r = self.order[self.No - self.err_rows]
        self.row_accumulator += popcount_vector(E[r, :])


cdef class ChainedE6MCC(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.per_output_evaluator = ChainedPerOutputMCC(Ne, No, cols, feature_order)

    cpdef reset(self):
        self.per_output_evaluator.reset()

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        self.per_output_evaluator.partial_evaluation(E, T)

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r
        cdef double[:] per_output

        per_output = self.per_output_evaluator.final_evaluation(E, T)

        # find earliest row with an error value
        for i in range(self.No):
            r = self.order[i]
            if self.per_output_evaluator.errors_exist[r]:
                return (-1.0 * (self.No - i - 1) + per_output[r]) / self.No
        return 0.0


cdef class ChainedE7(ChainedEvaluator):
    def __init__(self, size_t Ne, size_t No, size_t cols, size_t[:] feature_order):
        super().__init__(Ne, No, cols, feature_order)
        self.reset()

    cpdef reset(self):
        self.err_rows = 0

    cpdef partial_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T, size_t end_sub=0):
        cdef size_t i, r, c
        
        # first check if an earlier row now has an error value
        for i in range(self.err_rows, self.No):
            r = self.order[i]
            for c in range(self.cols):
                if E[r, c] > 0:
                    self.err_rows = self.No - i
                    return

    cpdef double final_evaluation(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        self.partial_evaluation(E, T)
        return self.Ne / self.divisor * self.err_rows 

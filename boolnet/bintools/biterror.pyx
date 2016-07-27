# cython: language_level=3
import boolnet.bintools.functions as fn
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt
from boolnet.bintools.bitcount cimport popcount_matrix, popcount_vector, floodcount_vector, floodcount_chunk
from boolnet.bintools.packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET, generate_end_mask
from boolnet.bintools.packing import packed_type


EVALUATORS = {
    fn.E1: StandardE1,
    fn.E2: StandardE2,
    fn.E3: StandardE3,
    fn.E4: StandardE4,
    fn.E5: StandardE5,
    fn.E6: StandardE6,
    fn.E7: StandardE7,
    fn.E1_MCC: MeanMCC,
    fn.E2_MCC: StandardE2MCC,
    fn.E6_MCC: StandardE6MCC,
    fn.CORRECTNESS: Correctness,
    fn.PER_OUTPUT_ERROR: PerOutputMean,
    fn.PER_OUTPUT_MCC: PerOutputMCC
}


cdef confusion(packed_type_t[:] errors, packed_type_t[:] target, size_t Ne,
               packed_type_t[:] TP_buf, packed_type_t[:] FP_buf, packed_type_t[:] FN_buf):
    # returns (TP, TN, FP, FN)
    cdef size_t TP, FP, FN
    cdef size_t blocks = errors.shape[0]
    for b in range(blocks):
        TP_buf[b]  = ~errors[b] &  target[b]
        FP_buf[b] =  errors[b] & ~target[b]
        FN_buf[b] =  errors[b] &  target[b]
                
    TP = popcount_vector(TP_buf)
    FP = popcount_vector(FP_buf)
    FN = popcount_vector(FN_buf)
    return TP, Ne - TP - FP - FN, FP, FN 


cdef double matthews_corr_coef(size_t TP, size_t TN, size_t FP, size_t FN):
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


cdef bint is_permutation(size_t[:] P, size_t No):
    return len(P) == No and min(P) == 0 and max(P) == No - 1 and len(np.unique(P)) == No


cdef class Evaluator:
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        if Ne % PACKED_SIZE == 0:
            self.cols = Ne // PACKED_SIZE
        else:
            self.cols = Ne // PACKED_SIZE + 1
        self.Ne = Ne
        self.No = No
        self.divisor = No * Ne
        # check feature_order is a valid permutation
        if not is_permutation(feature_order, No):
            raise ValueError('ChainedEvaluator - Error {} is not a size {} permutation!'.format(
                np.asarray(feature_order), No))
        self.order = np.array(feature_order)


cdef class PerOutputMCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.tp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fn_buffer = np.zeros(self.cols, dtype=packed_type)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        self.end_mask = generate_end_mask(Ne)
        self.errors_exist = np.zeros(No, dtype=np.uint8)

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, TP, TN, FP, FN
        cdef size_t actual_positives, actual_negatives, normaliser
        
        for i in range(self.No):
            TP, TN, FP, FN = confusion(
                E[i, :], T[i, :], self.Ne, self.tp_buffer,
                self.fp_buffer, self.fn_buffer)
            # keep in case a subclass wishes to use
            self.errors_exist[i] = (FP + FN) > 0
            self.mcc[i] = matthews_corr_coef(TP, TN, FP, FN)
        return self.mcc


cdef class PerOutputMean(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.accumulator = np.zeros(self.No, dtype=np.float64)
        self.divisor = Ne

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i

        for i in range(self.No):
            self.accumulator[i] = popcount_vector(E[i, :]) / self.divisor
        return self.accumulator


cdef class MeanMCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.per_output_evaluator = PerOutputMCC(Ne, No, feature_order)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        return sum(self.per_output_evaluator.evaluate(E, T)) / <double>self.No


cdef class Correctness(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)
        self.divisor = Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, c
        
        self.row_disjunction[:] = 0
        for i in range(self.No):
            for c in range(self.cols):
                self.row_disjunction[c] |= E[i, c]

        return 1.0 - popcount_vector(self.row_disjunction) / self.divisor


cdef class StandardE1(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef double result = 0.0
        return popcount_matrix(E) / self.divisor


cdef class StandardE2(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        cdef size_t i
        super().__init__(Ne, No, feature_order)
        # highest weight to earliest features - normalises to [0, 1]
        weights = np.arange(No, 0, -1, dtype=np.float64) / (0.5 * No * (No + 1.0))
        self.weight_vector = np.empty(No, dtype=np.float64)
        # reorder by feature order
        for i in range(No):
            self.weight_vector[i] = weights[self.order[i]]

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
            
        for i in range(self.No):
            result += popcount_vector(E[i, :]) * self.weight_vector[i] 

        return result / self.Ne


cdef class StandardE2MCC(StandardE2):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.per_output_evaluator = PerOutputMCC(Ne, No, feature_order)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
        cdef double[:] per_output

        per_output = self.per_output_evaluator.evaluate(E, T)

        for i in range(self.No):
            result += per_output[i] * self.weight_vector[i] 
        return result


cdef class StandardE3(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.row_disjunction = np.zeros(self.cols, dtype=packed_type)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, c
        cdef double result = 0.0
        
        self.row_disjunction[:] = 0
        for i in range(self.No):
            r = self.order[i]
            for c in range(self.cols):
                self.row_disjunction[c] |= E[r, c]
            result += popcount_vector(self.row_disjunction)
        return result / self.divisor


cdef class StandardE4(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum = 0
        cdef double result = 0.0

        for i in range(self.No):
            r = self.order[i]
            row_sum = max(row_sum, floodcount_vector(E[r, :], self.end_subtractor))
            result += row_sum

        return result / self.divisor


cdef class StandardE5(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.end_subtractor = self.cols * PACKED_SIZE - Ne % (self.cols * PACKED_SIZE)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        for i in range(self.No):
            r = self.order[i]
            row_sum = floodcount_vector(E[r, :], self.end_subtractor)
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
        return 0.0


cdef class StandardE6(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        
        # find earliest row with an error value
        for i in range(self.No):
            r = self.order[i]
            row_sum = popcount_vector(E[r, :])
            if row_sum > 0:
                return (self.Ne * (self.No - i - 1) + row_sum) / self.divisor
        return 0.0


cdef class StandardE6MCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, size_t[:] feature_order):
        super().__init__(Ne, No, feature_order)
        self.per_output_evaluator = PerOutputMCC(Ne, No, feature_order)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        cdef double[:] per_output

        per_output = self.per_output_evaluator.evaluate(E, T)

        # find earliest row with an error value
        for i in range(self.No):
            r = self.order[i]
            if self.per_output_evaluator.errors_exist[r]:
                return (-1.0 * (self.No - i - 1) + per_output[r]) / self.No
        return 0.0


cdef class StandardE7(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, c
        
        # find earliest row with an error value
        for i in range(self.No):
            r = self.order[i]
            for c in range(self.cols):
                if E[r, c] > 0:
                    return self.Ne / self.divisor * (self.No - i)
        return 0.0

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
    fn.E1:  (StandardE1, False),
    fn.E2L: (StandardE2, False), fn.E2M: (StandardE2, True),
    fn.E3L: (StandardE3, False), fn.E3M: (StandardE3, True),
    fn.E4L: (StandardE4, False), fn.E4M: (StandardE4, True),
    fn.E5L: (StandardE5, False), fn.E5M: (StandardE5, True),
    fn.E6L: (StandardE6, False), fn.E6M: (StandardE6, True),
    fn.E7L: (StandardE7, False), fn.E7M: (StandardE7, True),
    fn.E1_MCC: (MeanMCC, False),
    fn.E2L_MCC: (StandardE2MCC, False), fn.E2M_MCC: (StandardE2MCC, True),
    fn.E6L_MCC: (StandardE6MCC, False), fn.E6M_MCC: (StandardE6MCC, True),
    fn.ACCURACY: (Accuracy, False),
    fn.PER_OUTPUT_ERROR: (PerOutputMean, False),
    fn.PER_OUTPUT_MCC: (PerOutputMCC, False)
}


cdef confusion(packed_type_t[:] errors, packed_type_t[:] target, packed_type_t end_mask, size_t Ne,
               packed_type_t[:] TP_buffer, packed_type_t[:] FP_buffer, packed_type_t[:] FN_buffer):
    # returns (TP, TN, FP, FN)
    cdef size_t TP, FP, FN
    cdef size_t blocks = errors.shape[0]
    for b in range(blocks):
        TP_buffer[b]  = ~errors[b] &  target[b]
        FP_buffer[b] =  errors[b] & ~target[b]
        FN_buffer[b] =  errors[b] &  target[b]
    # ensure we do not count past Ne
    TP_buffer[-1] &= end_mask
    FP_buffer[-1] &= end_mask
    FN_buffer[-1] &= end_mask
                
    TP = popcount_vector(TP_buffer)
    FP = popcount_vector(FP_buffer)
    FN = popcount_vector(FN_buffer)
    return TP, Ne - TP - FP - FN, FP, FN 


cdef double matthews_correlation_coefficient(size_t TP, size_t TN, size_t FP, size_t FN):
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
        self.tp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fp_buffer = np.zeros(self.cols, dtype=packed_type)
        self.fn_buffer = np.zeros(self.cols, dtype=packed_type)
        self.TP = np.zeros(No, dtype=np.uintp)
        self.TN = np.zeros(No, dtype=np.uintp)
        self.FP = np.zeros(No, dtype=np.uintp)
        self.FN = np.zeros(No, dtype=np.uintp)
        self.mcc = np.zeros(self.No, dtype=np.float64)
        self.end_mask = generate_end_mask(Ne)

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef size_t actual_positives, actual_negatives, normaliser
        
        for i in range(self.No):
            self.TP[i], self.TN[i], self.FP[i], self.FN[i] = confusion(
                E[i, :], T[i, :], self.end_mask, self.Ne,
                self.tp_buffer, self.fp_buffer, self.fn_buffer)
            
            self.mcc[i] = matthews_correlation_coefficient(
                self.TP[i], self.TN[i], self.FP[i], self.FN[i])
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


cdef class StandardE2MCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        cdef size_t i 
        super().__init__(Ne, No, msb)
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)
        # normalise the weight vector
        for i in range(No):
            self.weight_vector[i] /= 0.5 * (No + 1.0) * No
        self.per_out_evaluator = PerOutputMCC(Ne, No, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
        cdef double[:] per_output

        per_output = self.per_output_evaluator.evaluate(E, T)
            
        for i in range(self.No):
            result += per_output[i] * self.weight_vector[i] 
        return result


cdef class StandardE6MCC(Evaluator):
    def __init__(self, size_t Ne, size_t No, bint msb):
        super().__init__(Ne, No, msb)
        self.per_out_evaluator = PerOutputMCC(Ne, No, msb)

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i, r, row_sum
        cdef double[:] per_output

        per_output = self.per_output_evaluator.evaluate(E, T)

        # find earliest row with an error value
        r = self.start
        for i in range(self.No):
            if self.per_output_evaluator.FP[i] + self.per_output_evaluator.FN[i] > 0:
                return (-1.0 * (self.No - i - 1) + per_output[r]) / self.No
            r += self.step
        return 0.0


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
        cdef size_t i
        super().__init__(Ne, No, msb)
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)
        # normalise the weight vector
        for i in range(No):
            self.weight_vector[i] /= 0.5 * (No + 1.0) * No

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T):
        cdef size_t i
        cdef double result = 0.0
            
        for i in range(self.No):
            result += popcount_vector(E[i, :]) * self.weight_vector[i] 

        return result / self.Ne


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

from BoolNet.metric_names import (
    E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
    E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT)
import numpy as np
from math import ceil
# CUDA
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

mod = SourceModule("""

__global__ void e1(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;

    int result = 0;
    for(int c = 0; c < num_cols; c++) {
        result += M[r*num_cols + c];
    }
    dest[r] = result;
}

__global__ void accuracy(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;

    int found = 0;
    for(int c = 0; c < num_cols; c++) {
        found = found || M[r*num_cols + c];
    }
    dest[r] = not found;
}

__global__ void e2_msb(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;

    int result = 0;
    for(int c = 0; c < num_cols; c++) {
        result += M[r*num_cols + c] * (c + 1);
    }
    dest[r] = result;
}

__global__ void e2_lsb(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;

    int result = 0;
    for(int c = 0; c < num_cols; c++) {
        result += M[r*num_cols + c] * (num_cols - c);
    }
    dest[r] = result;
}

__global__ void e3_msb(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;


    int found = 0;
    int result = 0;
    for(int c = num_cols - 1; c >= 0; c--) {
        found = found || M[r*num_cols + c];
        result += found;
    }
    dest[r] = result;
}

__global__ void e3_lsb(int *dest, char *M, int num_rows, int num_cols)
{
    const int r = threadIdx.x + blockIdx.x * blockDim.x;

    if(r >= num_rows)
        return;

    int found = 0;
    int result = 0;
    for(int c = 0; c < num_cols; c++) {
        found = found || M[r*num_cols + c];
        result += found;
    }
    dest[r] = result;
}

""")

gpu_e1 = mod.get_function('e1')
gpu_accuracy = mod.get_function('accuracy')
gpu_e2_msb = mod.get_function('e2_msb')
gpu_e2_lsb = mod.get_function('e2_lsb')
gpu_e3_msb = mod.get_function('e3_msb')
gpu_e3_lsb = mod.get_function('e3_lsb')

IMPLEMENTED_METRICS = [E1,
                       E2M,
                       E2L,
                       E3M,
                       E3L,
                       E7M,
                       E7L,
                       ACCURACY,
                       PER_OUTPUT]


# Many of the calculations in this method rely on error_matrix only being
# comprised of 1s and 0s
def metric_value_gpu(d_error_matrix, d_intermediate_matrix, Ne, No, metric):
    block_size = (16, 16, 1)
    grid_size = (ceil(Ne/16), ceil(No/16), 1)
    intermediate_gpuarray = gpuarray.GPUArray(shape=Ne, dtype=np.int32,
                                              gpudata=d_intermediate_matrix)
    Ne = np.int32(Ne)
    No = np.int32(No)
    weight_denominator = (No*(No+1))/2

    if metric == E1:
        gpu_e1(d_intermediate_matrix, d_error_matrix, Ne, No,
               grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne / No

    if metric == E2M:
        gpu_e2_msb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne / weight_denominator

    if metric == E2L:
        gpu_e2_lsb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne / weight_denominator

    if metric == E3M:
        gpu_e3_msb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne / No

    if metric == E3L:
        gpu_e3_lsb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne / No

    # E7 uses the kernel for e3
    if metric == E7M:
        gpu_e3_msb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.max(intermediate_gpuarray).get() / No

    if metric == E7L:
        gpu_e3_lsb(d_intermediate_matrix, d_error_matrix, Ne, No,
                   grid=grid_size, block=block_size)
        return gpuarray.max(intermediate_gpuarray).get() / No

    if metric == ACCURACY:
        gpu_accuracy(d_intermediate_matrix, d_error_matrix, Ne, No,
                     grid=grid_size, block=block_size)
        return gpuarray.sum(intermediate_gpuarray).get() / Ne

# ################# MULTI-VALUED METRICS ################# #

    if metric == PER_OUTPUT:
        error_gpuarray = gpuarray.GPUArray(shape=(Ne, No), dtype=np.byte,
                                           gpudata=d_error_matrix)
        result = np.empty(shape=No, dtype=np.int32)
        subset = gpuarray.arange(0, Ne*No, No, dtype=np.int32)
        for o in range(No):
            result[o] = gpuarray.subset_sum(
                subset+o, error_gpuarray, dtype=np.int32).get()
        return result / Ne

    raise ValueError('Invalid or unimplemented metric - {}'.format(metric))

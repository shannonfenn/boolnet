import collections
from math import ceil
from BoolNet.NetworkEvaluator import NetworkEvaluator
import BoolNet.BitErrorGPU as BitErrorGPU
import numpy as np
# CUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void evaluate_state(char* state, int* gates,
                               int Ni, int Ng, int Ne)
{
    // Basic map operation version, each thread evaluates a full network
    // this can probably be sped up but it would be much more complicated
    // also moving the gates over to shared memory might be worth it (maybe)

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if( id >= Ne )
        return;

    char* example = state + id*(Ni + Ng);
    for( int g=0; g<Ng; g++) {
        example[g+Ni] = not ( example[ gates[g*2] ] and example[ gates[g*2+1] ] );
    }
}

__global__ void evaluate_error(char* state, char* target, char* error,
                               int Ni, int Ng, int Ne, int No)
{
    // Basic map operation version, each thread evaluates a single entry
    // in the error matrix
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if( id >= No * Ne )
        return;

    int row = id / No;
    int col = id % No;
    error[row*No + col] = ( state[row*(Ni+Ng) + Ni + Ng - No + col] != target[row*No + col] );
}
  """)

eval_state = mod.get_function("evaluate_state")
eval_error = mod.get_function("evaluate_error")

DeviceMemory = collections.namedtuple('DeviceMemory', [
    'd_gates', 'd_activation', 'd_error', 'd_temp'])


class NetworkEvaluatorGPU(NetworkEvaluator):

    def __init__(self, inputs, target):
        super().__init__(inputs, target)
        # copy target to device
        self._d_target = cuda.to_device(self._target)
        # keep track of device memory per network
        self._device_states = []
        # block and thread sizes
        self._block_size = (512, 1, 1)
        self._grid_size_err = (ceil(self._target.size/512), 1, 1)

    def output_matrix(self, index):
        self.evaluate(index)
        # copy result matrices back to host - currently assumes this is
        # requested far less often than evaluate() is called
        # TODO: track if this has been done and only do so if not
        cuda.memcpy_dtoh(self._states[index].activation,
                         self._device_states[index].d_activation)
        return self._states[index].output

    def error_matrix(self, index):
        self.evaluate(index)
        # copy result matrices back to host - currently assumes this is
        # requested far less often than evaluate() is called
        # TODO: track if this has been done and only do so if not
        cuda.memcpy_dtoh(self._states[index].error,
                         self._device_states[index].d_error)
        return self._states[index].error

    def activation_matrix(self, index):
        self.evaluate(index)
        # copy result matrices back to host - currently assumes this is
        # requested far less often than evaluate() is called
        # TODO: track if this has been done and only do so if not
        cuda.memcpy_dtoh(self._states[index].activation,
                         self._device_states[index].d_activation)
        return self._states[index].activation

    def add_network(self, network):
        ''' Set up the activation and error matrices for a new network and add the
            resulting State struct to the states list.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        super().add_network(network)

        state = self._states[-1]
        Ne = self._inputs.shape[0]

        # allocate device arrays
        self._device_states.append(DeviceMemory(
            d_gates=cuda.mem_alloc(state.network.gates.nbytes),
            d_activation=cuda.mem_alloc(state.activation.nbytes),
            d_error=cuda.mem_alloc(state.error.nbytes),
            d_temp=cuda.mem_alloc(Ne * 4)))

    def remove_network(self, index):
        super().remove_network(index)
        for mem in self._device_states[index]:
            mem.free()
        self._device_states.pop(index)

    def remove_all_networks(self):
        self._states = []
        for device_state in self._device_states:
            for mem in device_state:
                mem.free()
        self._device_states = []

    def evaluate(self, index):
        ''' Evaluate the activation and error matrices for the selected network if it
            has been modified since the last evaluation. '''
        state = self._states[index]
        device_state = self._device_states[index]

        if not state.network._evaluated:
            Ne = np.int32(self._inputs.shape[0])
            Ni = np.int32(self._inputs.shape[1])
            No = np.int32(self._target.shape[1])
            Ng = np.int32(state.network.Ng)

            # copy gates and matrices to GPU
            cuda.memcpy_htod(device_state.d_gates, state.network.gates)
            cuda.memcpy_htod(device_state.d_activation, state.activation)

            # perform calculations on GPU
            eval_state(device_state.d_activation, device_state.d_gates,
                       Ni, Ng, Ne, block=self._block_size)

            eval_error(device_state.d_activation, self._d_target,
                       device_state.d_error, Ni, Ng, Ne, No,
                       grid=self._grid_size_err, block=self._block_size)

            state.network._evaluated = True

    def metric_value(self, index, metric):
        self.evaluate(index)
        Ne = np.int32(self._target.shape[0])
        No = np.int32(self._target.shape[1])
        return BitErrorGPU.metric_value_gpu(
            self._device_states[index].d_error, self._device_states[index].d_temp,
            Ne, No, metric)

from BoolNet.BooleanNetwork import BooleanNetwork
import numpy as np


class RandomBooleanNetwork(BooleanNetwork):

    def __init__(self, initial_gates, Ni, No, transfer_functions):
        self._transfer_functions = np.array(
            transfer_functions, dtype=np.uint8, copy=True)
        super().__init__(initial_gates, Ni, No)

    def _check_invariants(self):
        super()._check_invariants()
        if self._transfer_functions.shape != (self.Ng,):
            raise ValueError('Invalid transfer function matrix shape: {}'.format(
                self._transfer_functions.shape))

    def __str__(self):
        return super().__str__() + '\ntransfer functions:\n{}'.format(self._transfer_functions)

    @property
    def transfer_functions(self):
        return self._transfer_functions

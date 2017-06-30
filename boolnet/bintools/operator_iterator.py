def _get_operator_evaluator(op, Nb):
    divisor = 2**Nb
    if op == 'zero':
        return lambda x: 0
    elif op == 'not':
        return lambda x: ~x
    elif op == 'unary_and':
        all_ones = divisor - 1
        return lambda x: x == all_ones
    elif op == 'unary_or':
        return lambda x: x != 0
    elif op == 'and':
        return lambda x: int(x // divisor) & int(x % divisor)
    elif op == 'or':
        return lambda x: int(x // divisor) | int(x % divisor)
    elif op == 'xor':
        return lambda x: int(x // divisor) ^ int(x % divisor)
    elif op == 'add':
        return lambda x: int(x // divisor) + int(x % divisor)
    elif op == 'sub':
        return lambda x: int((int(x // divisor) - int(x % divisor)) % divisor)
    elif op == 'mul':
        return lambda x: int(x // divisor) * int(x % divisor)
    else:
        raise ValueError('Invalid operator: {}'.format(op))


num_operands = {
    'zero': 1,
    'and': 2,
    'or': 2,
    'unary_and': 1,
    'unary_or': 1,
    'add': 2,
    'sub': 2,
    'mul': 2,
}


def operator_example_iterator(operator, Nb, indices, exclude):
    Ni = num_operands[operator] * Nb
    if exclude:
        return OpExcIterator(operator, Nb, 2**Ni, indices)
    else:
        return OpIncIterator(operator, Nb, indices)


class OpIterator:
    def __init__(self, op, Nb, Ne):
        if Nb == 0:
            raise ValueError('Zero bitwidth not allowed.')
        self.func = _get_operator_evaluator(op, Nb)
        self.remaining = Ne

    def __iter__(self):
        return self

    def __len__(self):
        return self.remaining


class OpIncIterator(OpIterator):
    def __init__(self, op, Nb, include_list):
        super().__init__(op, Nb, len(include_list))
        self.include_iter = iter(include_list)

    def __next__(self):
        inp = next(self.include_iter)
        out = self.func(inp)
        self.remaining -= 1
        return inp, out


class OpExcIterator(OpIterator):
    def __init__(self, op, Nb, total_elements, exclude_list):
        if total_elements < len(exclude_list):
            raise ValueError(
                'Exclude list larger ({}) than total_elements ({})'.format(
                    len(exclude_list), total_elements))
        super().__init__(op, Nb, total_elements - len(exclude_list))
        self.total_elements = total_elements
        # sorted indices are required
        self.ex_iter = iter(sorted(exclude_list))
        self.index = 0
        try:
            self.ex_index = next(self.ex_iter)
            self._sync()
        except StopIteration:
            self.ex_index = self.total_elements

    def _sync(self):
        if self.ex_index < self.total_elements:
            try:
                while self.index == self.ex_index:
                    self.index += 1
                    self.ex_index = next(self.ex_iter)
            except StopIteration:
                self.ex_index = self.total_elements

    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        inp = self.index
        out = self.func(inp)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out

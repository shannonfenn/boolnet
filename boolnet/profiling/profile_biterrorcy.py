import numpy as np
from timeit import timeit
import json
import yaml
from progress.bar import Bar
import argparse
import sys
from os.path import dirname, realpath, join
import os
sys.path.append(os.path.expanduser('~/HMRI/code/python/'))

from BoolNet.function_names import all_functions, function_name, function_value

from BoolNet.Packing import packmat, packed_type, generate_end_mask


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Rough profiling of high cost error ops')
#     parser.add_argument('--file',
#                         type=argparse.FileType('r'),
#                         default='profiling/speed_test_small.json')
#     parser.add_argument('--evaluator', default='cy', choices=['py', 'cy', 'gpu'])

#     return parser.parse_args()


def eval_metric(params, metric):
    # force reevaluation
    E, E_scratch, Ne, end_mask = params
    return function_value(E, E_scratch, Ne, end_mask, metric)


def tests(metrics, param_sets, repeats):
    tests = dict()

    for name, params in param_sets.items():
        for metric in metrics:
            tests[name + ' - ' + function_name(metric)] = (
                'eval_metric(params, {})'.format(metric.raw_str()),
                'from __main__ import eval_metric, params, {}'.format(metric.raw_str()),
                repeats)

    return tests


def make_params(f):
    E_unpacked = np.array(json.load(f), dtype=packed_type)
    Ne, _ = E_unpacked.shape
    E = packmat(E_unpacked)
    E_scratch = np.zeros_like(E)
    end_mask = generate_end_mask(Ne)
    return (E, E_scratch, Ne, end_mask)


if __name__ == '__main__':
    # args = parse_arguments()

    param_sets = dict()
    with open('matrix_small.json') as f:
        param_sets['small'] = make_params(f)
    with open('matrix_large.json') as f:
        param_sets['large'] = make_params(f)

    tests = tests(list(all_functions()), param_sets, 100)

    bar = Bar('Analysing', max=len(tests))

    results = dict()
    for name, (test, setup, repeats) in tests.items():
        results[name] = min(timeit(test, setup=setup, number=repeats) / repeats for i in range(3))
        bar.next()

    bar.finish()

    print(yaml.dump(results, default_flow_style=False))

# ######## CODE USED TO GENERATE ABOVE FILE ########## #
# inputs = np.random.randint(0, 2, (1024, 16))
# target = np.random.randint(0, 2, (1024, 16))
# gates = np.zeros((1000, 2), dtype=int)
# for g in range(len(gates)):
#     gates[g] = np.random.randint(0, g+16, 2)
# test = {'inputs': inputs.tolist(),
#         'target': target.tolist(),
#         'gates' : gates.tolist()}
# with open('speed_test_net.json', 'w') as fp:
#     json.dump(test, fp)

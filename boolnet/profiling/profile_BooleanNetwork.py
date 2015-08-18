import numpy as np
from timeit import timeit
import json
import yaml
from progress.bar import Bar
import argparse
import sys
from os.path import dirname, realpath, join
import os
sys.path.append(os.path.expanduser('~/HMRI/Code/Python/'))

import pyximport
pyximport.install(build_dir=join(dirname(realpath(__file__)), '.pyxbld'))

from BoolNet.NetworkEvaluator import NetworkEvaluator
from BoolNet.NetworkEvaluatorCython import NetworkEvaluatorCython
from BoolNet.function_names import all_functions, function_name, Metric
from BoolNet.BooleanNetwork import BooleanNetwork
from BoolNet.RandomBooleanNetwork import RandomBooleanNetwork
from BoolNet.Packing import pack_bool_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(description='Rough profiling of high cost boolnet ops')
    parser.add_argument('--file',
                        type=argparse.FileType('r'),
                        default='profiling/speed_test_small.json')
    parser.add_argument('--evaluator', default='cy', choices=['py', 'cy', 'gpu'])

    return parser.parse_args()


def error_per_output(evaluator, index):
    # force reevaluation
    evaluator.network(index).force_reevaluation()
    return evaluator.error_per_output(index)


def function_value(evaluator, index, metric):
    # net.force_reevaluation()
    return evaluator.function_value(index, metric)


def truth_table(evaluator, index):
    # force reevaluation not needed
    return evaluator.truth_table(index)


def rand_move_revert(net):
    net.move_to_neighbour(net.random_move())
    net.revert_move()


def set_mask(net, sourceable, changeable):
    net.set_mask(sourceable, changeable)


def evaluate(evaluator, index):
    # evaluator.network(index).force_reevaluation()
    evaluator.evaluate(index)


def move_evaluate(net, evaluator, index, metric):
    net.move_to_neighbour(net.random_move())
    return evaluator.function_value(index, metric)


def tests(metrics, repeats):
    tests = {
        'move then revert (nand)': ('rand_move_revert(net0)',
                                    'from __main__ import rand_move_revert, net0',
                                    repeats),
        'move then revert (random)': ('rand_move_revert(net1)',
                                      'from __main__ import rand_move_revert, net1',
                                      repeats),
        'set mask (nand)': ('set_mask(net0, sourceable, changeable)',
                            'from __main__ import set_mask, net0, changeable, sourceable',
                            repeats),
        'set mask (random)': ('set_mask(net1, sourceable, changeable)',
                              'from __main__ import set_mask, net1, changeable, sourceable',
                              repeats),
        'evaluate (nand)': ('evaluate(evaluator, 0)',
                            'from __main__ import evaluate, evaluator; ' +
                            'evaluator.network(0).force_reevaluation()',
                            repeats),
        'evaluate (random)': ('evaluate(evaluator, 1)',
                              'from __main__ import evaluate, evaluator; ' +
                              'evaluator.network(1).force_reevaluation()',
                              repeats),
        # 'truth table': ('truth_table(net)',
        #                 'from __main__ import truth_table, net',
        #                 3)  # this is a slow operation
    }

    for metric in metrics:
        tests[function_name(metric) + ' (nand)'] = (
            'function_value(evaluator, 0, {})'.format(metric.raw_str()),
            'from __main__ import function_value, {}, evaluator'.format(metric.raw_str()),
            repeats)
        tests[function_name(metric) + ' (random)'] = (
            'function_value(evaluator, 1, {})'.format(metric.raw_str()),
            'from __main__ import function_value, {}, evaluator'.format(metric.raw_str()),
            repeats)

    for metric in [Metric.E1, Metric.E6_LSB]:
        tests['move then eval {} (nand)'.format(function_name(metric))] = (
            'function_value(evaluator, 0, {})'.format(metric.raw_str()),
            'from __main__ import function_value, {}, evaluator'.format(metric.raw_str()),
            repeats)
        tests['move then eval {} (random)'.format(function_name(metric))] = (
            'function_value(evaluator, 1, {})'.format(metric.raw_str()),
            'from __main__ import function_value, {}, evaluator'.format(metric.raw_str()),
            repeats)

    return tests

if __name__ == '__main__':
    args = parse_arguments()

    case = json.load(args.file)

    if args.evaluator == 'gpu':
        from BoolNet.NetworkEvaluatorGPU import NetworkEvaluatorGPU
        from BoolNet.BitErrorGPU import IMPLEMENTED_METRICS
        evaluator_class = NetworkEvaluatorGPU
        metrics = IMPLEMENTED_METRICS
    elif args.evaluator == 'cy':
        evaluator_class = NetworkEvaluatorCython
        metrics = list(all_functions())
    else:
        evaluator_class = NetworkEvaluator
        metrics = list(all_functions())

    tests = tests(metrics, 100)

    inputs = np.array(case['inputs'], dtype=np.byte)
    target = np.array(case['target'], dtype=np.byte)
    gates = np.array(case['gates'], dtype=np.uint32)
    Ng = len(gates)

    Ne, Ni = inputs.shape
    _, No = target.shape

    inputs = pack_bool_matrix(inputs)
    target = pack_bool_matrix(target)

    evaluator = evaluator_class(inputs, target, Ne)

    net0 = BooleanNetwork(gates, Ni, No)
    net1 = RandomBooleanNetwork(gates, Ni, No, np.random.randint(16, size=Ng))
    evaluator.add_network(net0)
    evaluator.add_network(net1)

    changeable = list(range(int(Ng/4), int(Ng/2)))
    sourceable = list(range(int(Ng/4)))

    results = dict()
    results['test file'] = args.file.name

    bar = Bar('Analysing', max=len(tests))

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

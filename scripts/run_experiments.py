#! /usr/bin/env python

import argparse
import pickle
import json
import gzip

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_single_experiment(expfile, verbose):
    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))
    result = learn_bool_net(task, verbose)
    result['id'] = task['id']
    return result


def process_single_experiments(expfile, verbose):
    resultfile = expfile + '.json'
    result = run_single_experiment(expfile, args.verbose)
    with open(resultfile, 'w') as ostream:
        json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                  separators=(',', ':'))


def process_multiple_experiments(explistfile, verbose):
    resultfile = explistfile + '.json'
    with open(explistfile) as tasks, open(resultfile, 'w') as ostream:
        for line in tasks:
            result = run_single_experiment(line.strip(), verbose)
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                      separators=(',', ':'))
            ostream.write('\n')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        process_multiple_experiments(args.experiment, args.verbose)
    elif args.experiment.endswith('.exp'):
        process_single_experiments(args.experiment, args.verbose)
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()

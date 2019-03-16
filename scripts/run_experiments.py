#! /usr/bin/env python

import argparse
import pickle
import json
import gzip
import os
import sys

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_single_experiment(expfile):
    task = pickle.load(gzip.open(expfile, 'rb'))
    result = learn_bool_net(task)
    result['id'] = task['id']
    return result


def process_single_experiment(expfile):
    resultfile = expfile + '.json'
    result = run_single_experiment(expfile)
    with open(resultfile, 'w') as ostream:
        json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                  separators=(',', ':'))


def process_multiple_experiments(explistfile):
    resultfile = explistfile + '.json'
    with open(explistfile) as tasks, open(resultfile, 'w', 1) as ostream:
        for line in tasks:
            try:
                result = run_single_experiment(line.strip())
            except ValueError as err:
                print(err, file=sys.stderr)
            else:
                json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                          separators=(',', ':'))
                ostream.write('\n')
                ostream.flush()
                os.fsync(ostream.fileno())


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        process_multiple_experiments(args.experiment)
    elif args.experiment.endswith('.exp'):
        process_single_experiment(args.experiment)
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()

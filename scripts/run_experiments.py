#! /usr/bin/env python

import argparse
import pickle
import json
import gzip

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_single_experiment(expfile):
    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))
    result = learn_bool_net(task)
    result['id'] = task['id']
    return result


def process_single_experiments(expfile):
    resultfile = expfile + '.json'
    result = run_single_experiment(expfile)
    with open(resultfile, 'w') as ostream:
        json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                  separators=(',', ':'))


def process_multiple_experiments(explistfile):
    resultfile = explistfile + '.json'
    with open(explistfile) as tasks, open(resultfile, 'w') as ostream:
        for line in tasks:
            result = run_single_experiment(line.strip())
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                      separators=(',', ':'))
            ostream.write('\n')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        process_multiple_experiments(args.experiment)
    elif args.experiment.endswith('.exp'):
        process_single_experiments(args.experiment)
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()

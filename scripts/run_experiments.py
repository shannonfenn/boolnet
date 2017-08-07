import argparse
import pickle
import json
import gzip
from os.path import splitext

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_multiple_experiments(explistfile, verbose):
    resultfile = explistfile + '.json'
    with open(explistfile) as tasks, open(resultfile, 'w') as ostream:
        for i, line in enumerate(tasks):
            result = run_single_experiment(line.strip(), verbose)
            ostream.write('[' if i == 0 else '\n,')
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder)
        ostream.write(']\n')


def run_single_experiment(expfile, verbose):
    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))
    result = learn_bool_net(task, verbose)
    result['id'] = task['id']
    return result


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        run_multiple_experiments(args.experiment, args.verbose)
    elif args.experiment.endswith('.exp'):
        result = run_single_experiment(args.experiment, args.verbose)
        resultfile = args.experiment + '.json'
        with open(resultfile, 'w') as ostream:
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder)
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()

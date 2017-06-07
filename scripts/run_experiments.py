import argparse
import pickle
import json
import gzip
from os.path import splitext

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_multiple_experiments(explistfile):
    with open(explistfile) as f:
        for line in f:
            run_single_experiment(line)


def run_single_experiment(expfile):
    resultfile = splitext(expfile)[0] + '.json'

    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))

    result = learn_bool_net(task)

    with open(resultfile, 'w') as stream:
        json.dump(result, stream, cls=NumpyAwareJSONEncoder)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        run_multiple_experiments(args.experiment)
    elif args.experiment.endswith('.exp'):
        run_single_experiment(args.experiment)
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()

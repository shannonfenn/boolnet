import argparse
import pickle
import json
import gzip
from os.path import splitext

from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools.learn_boolnet import learn_bool_net


def run_single_experiment(expfile):
    resultfile = splitext(expfile)[0] + '.json'

    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))

    result = learn_bool_net(task)

    with open(resultfile, 'w') as stream:
        json.dump(result, stream, cls=NumpyAwareJSONEncoder)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str, help='.exp file to run.')
    args = parser.parse_args()

    run_single_experiment(args.experiment)


if __name__ == '__main__':
    main()

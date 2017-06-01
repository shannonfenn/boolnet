import pickle
import argparse
import json
import gzip
from os.path import splitext
from boolnet.exptools.learn_boolnet import learn_bool_net
from boolnet.exptools.config_tools import ExperimentJSONEncoder


def main(expfile):
    resultfile = splitext(expfile)[0] + '.json'

    # with open(expfile, 'rb') as f:
    with gzip.open(expfile, 'rb') as f:
        task = pickle.load(f)

    result = learn_bool_net(task)

    with open(resultfile, 'w') as stream:
        json.dump(result, stream, cls=ExperimentJSONEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str, help='.exp file to run.')
    args = parser.parse_args()

    main(args.experiment)


# combine script

# write "["
# cat first subfile
# cat each subsequent subfile prepended with "," and ended with "\n"
# write "]"

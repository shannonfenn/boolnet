import pickle
import argparse
import json
from boolnet.exptools.learn_boolnet import learn_bool_net
from boolnet.exptools.config_tools import ExperimentJSONEncoder


def main(directory, index):
    # use streams instead

    expfile = '{}/working/{}.exp'.format(directory, index)
    resultfile = '{}/working/{}.json'.format(directory, index)

    with open(expfile, 'rb') as f:
        task = pickle.load(f)

    ## Prep data here

    result = learn_bool_net(task)

    with open(resultfile, 'w') as stream:
        json.dump(result, stream, cls=ExperimentJSONEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('exp_directory', type=str,
                        help='experiment result directory.')
    parser.add_argument('index', type=int,
                        help='experiment index to run.')
    args = parser.parse_args()
    main(args.exp_directory, args.index)

# combine script

# write "["
# cat first subfile
# cat each subsequent subfile prepended with "," and ended with "\n"
# write "]"

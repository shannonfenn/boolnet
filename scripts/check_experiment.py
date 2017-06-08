import re
import glob
import argparse
from os.path import abspath, expanduser, isdir, splitext
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = abspath(expanduser(directory))
    if isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid path'.format(directory))


def get_remaining_experiments(directory):
    exp_iter = glob.iglob('{}/working/*.exp'.format(directory))
    json_iter = glob.iglob('{}/working/*.json'.format(directory))

    all_exp = set(splitext(f)[0] for f in exp_iter)
    all_json = set(splitext(f)[0] for f in json_iter)
    remaining = natsorted(f + '.exp' for f in all_exp - all_json)
    print('\n'.join(remaining))


def get_failed_experiments(directory):
    failed_pattern = re.compile('trg_error": 0\.0(,|\})')
    json_iter = glob.iglob('{}/working/*.json'.format(directory))
    for fname in json_iter:
        with open(fname, 'r') as f:
            if failed_pattern.search(f.read()) is None:
                print(splitext(f)[0] + '.exp')


def main():
    parser = argparse.ArgumentParser(
        description='Tools for filtering experiment (.exp/.json) files')
    parser.add_argument('dir', type=directory_type)

    subparsers = parser.add_subparsers(help='commands', dest='command')

    parser_remaining = subparsers.add_parser('r')
    parser_remaining.set_defaults(func=get_remaining_experiments)

    parser_failed = subparsers.add_parser('f')
    parser_failed.set_defaults(func=get_failed_experiments)

    args = parser.parse_args()

    args.func(args.dir)

if __name__ == '__main__':
    main()

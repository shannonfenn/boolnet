import re
import glob
import argparse
from os.path import abspath, expanduser, isdir, splitext

FAILED_PATTERN = re.compile('trg_error": 0\.0(,|\})')

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
    remaining = [f + '.exp' for f in all_exp - all_json]
    for f in remaining:
        print(f)


def get_failed_experiments(directory):
    json_iter = glob.iglob('{}/working/*.json'.format(directory))
    for f in json_iter:
        if FAILED_PATTERN.search(open(f, 'r').read()) is None:
            print(splitext(f)[0] + '.exp')


def main():
    parser = argparse.ArgumentParser(
        description='Tools for filtering experiment (.exp/.json) files')

    subparsers = parser.add_subparsers(help='commands', dest='command')

    parser_remaining = subparsers.add_parser('r')
    parser_remaining.add_argument('dir', type=directory_type)
    parser_remaining.set_defaults(func=get_remaining_experiments)

    parser_failed = subparsers.add_parser('f')
    parser_failed.add_argument('dir', type=directory_type)
    parser_failed.set_defaults(func=get_failed_experiments)

    args = parser.parse_args()

    args.func(args.dir)

if __name__ == '__main__':
    main()

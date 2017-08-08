import re
import argparse
from os.path import abspath, expanduser, isdir, splitext
from natsort import natsorted


def get_non_memorised_experiments(all_files, pattern):
    failed = []
    for fname in all_files:
        with open(splitext(fname)[0] + '.json', 'r') as f:
            if pattern.search(f.read()) is None:
                failed.append(fname.strip())
    failed = natsorted(failed)
    print('\n'.join(failed))


def get_memorised_experiments(all_files, pattern):
    memorised = []
    for fname in all_files:
        with open(splitext(fname)[0] + '.json', 'r') as f:
            if pattern.search(f.read()) is not None:
                memorised.append(fname.strip())
    memorised = natsorted(memorised)
    print('\n'.join(memorised))


def main():
    parser = argparse.ArgumentParser(
        description='For checking lists of experiment (.exp/.json) files')

    subparsers = parser.add_subparsers(help='commands', dest='command')

    parser_failed = subparsers.add_parser('not')
    parser_failed.add_argument('list', type=argparse.FileType())
    parser_failed.set_defaults(func=get_non_memorised_experiments)

    parser_succeeded = subparsers.add_parser('mem')
    parser_succeeded.add_argument('list', type=argparse.FileType())
    parser_succeeded.set_defaults(func=get_memorised_experiments)

    args = parser.parse_args()

    pattern = re.compile('trg_err": 0\.0(,|\})')

    args.func(args.list, pattern)

if __name__ == '__main__':
    main()

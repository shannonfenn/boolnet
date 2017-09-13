#! /usr/bin/env python

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
    all_exp = glob.glob('{}/working/*.exp'.format(directory))
    all_json = glob.glob('{}/working/*.json'.format(directory))

    all_json = set(f[:-5] for f in all_json)

    remaining = natsorted(f for f in all_exp if f[:-4] not in all_json)

    print('\n'.join(remaining))


def get_non_memorised_experiments(directory):
    pattern = re.compile('trg_err": 0\.0(,|\})')
    all_json = glob.glob('{}/working/*.json'.format(directory))
    failed = []
    for fname in all_json:
        with open(fname, 'r') as f:
            if pattern.search(f.read()) is None:
                failed.append(splitext(fname)[0] + '.exp')
    failed = natsorted(failed)
    print('\n'.join(failed))


def get_memorised_experiments(directory):
    pattern = re.compile('trg_err": 0\.0(,|\})')
    all_json = glob.glob('{}/working/*.json'.format(directory))
    memorised = []
    for fname in all_json:
        with open(fname, 'r') as f:
            if pattern.search(f.read()) is not None:
                memorised.append(splitext(fname)[0] + '.exp')
    memorised = natsorted(memorised)
    print('\n'.join(memorised))


def summary(directory):
    pattern = re.compile('trg_err": 0\.0(,|\})')
    all_exp = glob.glob('{}/working/*.exp'.format(directory))
    all_json = glob.glob('{}/working/*.json'.format(directory))
    num_remaining = len(all_exp) - len(all_json)
    num_failed = 0
    for fname in all_json:
        with open(fname, 'r') as f:
            if pattern.search(f.read()) is None:
                num_failed += 1
    print('remaining: {} memorised: {} not-memorised: {}'.format(
        num_remaining, len(all_json) - num_failed, num_failed))


def main():
    parser = argparse.ArgumentParser(
        description='Tools for filtering experiment (.exp/.json) files')

    subparsers = parser.add_subparsers(help='commands', dest='command')

    parser_remaining = subparsers.add_parser('rem')
    parser_remaining.add_argument('dir', type=directory_type)
    parser_remaining.set_defaults(func=get_remaining_experiments)

    parser_failed = subparsers.add_parser('not')
    parser_failed.add_argument('dir', type=directory_type)
    parser_failed.set_defaults(func=get_non_memorised_experiments)

    parser_succeeded = subparsers.add_parser('mem')
    parser_succeeded.add_argument('dir', type=directory_type)
    parser_succeeded.set_defaults(func=get_memorised_experiments)

    parser_succeeded = subparsers.add_parser('sum')
    parser_succeeded.add_argument('dir', type=directory_type)
    parser_succeeded.set_defaults(func=summary)

    args = parser.parse_args()

    args.func(args.dir)

if __name__ == '__main__':
    main()

#! /usr/bin/env python

import argparse
import glob
import itertools
import os
from os.path import isdir, expanduser, abspath, join
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = abspath(expanduser(directory))
    if isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid directory.'.format(directory))


def strided(l, n):
    sublists = [[] for i in range(n)]
    for i, item in enumerate(l):
        sublists[i % n].append(item)
    return sublists


def bundle(dir, num=None, experiments=None):
    if experiments is None:
        experiments = natsorted(glob.glob('{}/tasks/*.exp'.format(dir)))

    if not experiments:
        print("No experiments found.")
        return

    num = min(num, len(experiments))

    bundles = strided(experiments, num)

    # find first non-existant */run_<int>/
    dir_generator = (join(dir, str(i))
                     for i in itertools.count())
    run_dir = next(directory
                   for directory in dir_generator
                   if not isdir(directory))
    os.makedirs(run_dir)

    for i, bundle in enumerate(bundles):
        fname = join(run_dir, '{}.explist'.format(i))
        with open(fname, 'w') as f:
            f.write('\n'.join(bundle))
            f.write('\n')

    print('{} bundles created.'.format(len(bundles)))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=directory_type)
    parser.add_argument('--num', '-n', type=int, default=7500)
    parser.add_argument('--infile', '-i', type=argparse.FileType(),
                        help='list of experiments (.exp) to run')

    args = parser.parse_args()

    if not (1 <= args.num <= 7500):
        parser.error('--num must be in [1..7500].')

    if args.infile:
        experiments = [line.strip()
                       for line in args.infile
                       if line.strip()]
        args.infile.close()
    else:
        experiments = None
    bundle(args.dir, args.num, experiments)


if __name__ == '__main__':
    main()

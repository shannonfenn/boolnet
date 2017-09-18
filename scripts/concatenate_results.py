#! /usr/bin/env python

import re
import glob
import argparse
import os.path
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = os.path.abspath(os.path.expanduser(directory))
    if os.path.isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid directory.'.format(directory))


def concatenate(partials, outstream):
    lead_char = '['
    for fname in partials:
        with open(fname, 'r') as f:
            for line in f:
                # extract the record
                match = re.search(pattern='\{.*\}', string=line)
                if match:
                    outstream.write(lead_char)
                    outstream.write(match.group())
                    outstream.write('\n')
                    lead_char = ','
    outstream.write(']\n')


def main():
    parser = argparse.ArgumentParser(description='joins <run>/*.json')
    parser.add_argument('dir', type=directory_type)
    parser.add_argument('run', nargs='?', type=str, default='0')
    parser.add_argument('--outfile', '-o', type=argparse.FileType('x'),
                        help=('output file, default: <dir>/<run>.json'))
    args = parser.parse_args()

    run_dir = os.path.join(args.dir, args.run)

    if not os.path.isdir(run_dir):
        raise ValueError('{} doesn\'t exist'.format(run_dir))
    if not args.outfile:
        fname = os.path.join(args.dir, args.run + '.json')
        args.outfile = open(fname, 'x')

    partials = glob.glob(os.path.join(run_dir, '*.json'))
    partials = natsorted(partials)
    concatenate(partials, args.outfile)

if __name__ == '__main__':
    main()

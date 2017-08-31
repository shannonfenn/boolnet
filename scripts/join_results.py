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


def parse_args():
    parser = argparse.ArgumentParser(description='joins working/*.json')
    parser.add_argument('dir', type=directory_type)
    parser.add_argument('--outfile', '-o', type=argparse.FileType('x'),
                        help=('output file, default: <dir>/results.json'))
    args = parser.parse_args()
    if not args.outfile:
        args.outfile = open(os.path.join(args.dir, 'results.json'), 'x')
    return args


def main():
    args = parse_args()
    partials = glob.glob(os.path.join(args.dir, 'working', '*.json'))
    partials = natsorted(partials)
    concatenate(partials, args.outfile)

if __name__ == '__main__':
    main()

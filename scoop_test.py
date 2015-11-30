import time
import os
import sys
import argparse
from scoop.futures import map_as_completed


def print_details(verbose):
    # Check version
    s = 'HOSTNAME: {}'.format(os.environ.get("HOSTNAME"))
    s += '\nCIBM_HOME: {}'.format(os.environ.get("CIBM_HOME"))
    s += '\npy ver: {}'.format(sys.version_info)
    s += '\npy loc: {}'.format(sys.executable)
    if verbose:
        s += '\nenviron: {}'.format(os.environ)
    time.sleep(3)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # num_servers = 14 + 6
    num_servers = 3
    # uses unordered map to ensure results are dumped as soon as available
    results = list(map_as_completed(
        print_details, [args.verbose]*(num_servers + 1)))

    for r in results:
        print(r)

if __name__ == '__main__':
    main()

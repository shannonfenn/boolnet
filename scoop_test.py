import time
import os
import sys
import argparse
import socket
from scoop.futures import map_as_completed


def print_details(verbose):
    # Check version
    s = ('socket.gethostname(): {}\n$HOST: {}\n$HOSTNAME: {}\n$CIBM_HOME: {}\n'
         'py ver: {}\npy loc: {}\n').format(
        socket.gethostname(),
        os.environ.get("HOST"),
        os.environ.get("HOSTNAME"),
        os.environ.get("CIBM_HOME"),
        sys.version_info,
        sys.executable)

    if verbose:
        s += 'environ: {}\n'.format(os.environ)
    time.sleep(3)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    num_servers = 14 + 6
    # num_servers = 3
    # uses unordered map to ensure results are dumped as soon as available
    results = list(map_as_completed(
        print_details, [args.verbose]*(num_servers + 1)))

    with open('log.txt', 'w') as f:
        for r in results:
            f.write(r + '\n')

if __name__ == '__main__':
    main()

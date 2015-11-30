import time
import os
import sys
import argparse
import socket
from scoop.futures import map_as_completed


def print_details(sleeptime):
    # Check version
    time.sleep(sleeptime)

    return {
        'socket.gethostname()': socket.gethostname(),
        '$HOST': os.environ.get("HOST"),
        '$HOSTNAME': os.environ.get("HOSTNAME"),
        '$CIBM_HOME': os.environ.get("CIBM_HOME"),
        'py ver': sys.version_info,
        'py loc': sys.executable,
        'environ': os.environ}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    sleeptime = 3

    num_servers = 14 + 6
    # num_servers = 3
    # uses unordered map to ensure results are dumped as soon as available
    results = list(map_as_completed(
        print_details, [sleeptime]*(num_servers + 1)))

    with open('log.txt', 'w') as f:
        for r in results:
            if not args.verbose:
                r.pop('environ')
            f.write(r + '\n')

if __name__ == '__main__':
    main()

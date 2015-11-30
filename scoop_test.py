import time
import os
import sys
from scoop.futures import map_as_completed


def print_details(idx):
    # Check version
    s = 'idx: {}'.format(idx)
    s += '\nenviron: {}'.format(os.environ)
    s += '\nHOSTNAME: {}'.format(os.environ.get("HOSTNAME"))
    s += '\nCIBM_HOME: {}'.format(os.environ.get("CIBM_HOME"))
    s += '\npy ver: {}'.format(sys.version_info)
    s += '\npy loc: {}'.format(sys.executable)
    time.sleep(3)
    return s


def main():
    # num_servers = 14 + 6
    num_servers = 3
    # uses unordered map to ensure results are dumped as soon as available
    results = list(map_as_completed(print_details, range(num_servers + 1)))

    for r in results:
        print(r)

if __name__ == '__main__':
    main()

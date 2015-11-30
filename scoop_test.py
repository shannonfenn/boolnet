import time
import os                           # for mkdir
import sys                          # for path, exit
import scoop                        # for distributed parallellism


def print_details(idx):
    # Check version
    print(idx)
    print(os.environ.get("HOSTNAME"))
    print(os.environ.get("CIBM_HOME"))
    print(sys.version_info)
    print(sys.executable)
    time.sleep(3)
    return idx


def main():
    # num_servers = 14 + 6
    num_servers = 3
    # uses unordered map to ensure results are dumped as soon as available
    for i in scoop.futures.map(print_details, range(num_servers + 1)):
        pass


if __name__ == '__main__':
    main()

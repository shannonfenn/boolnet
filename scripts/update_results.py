import json
import argparse


def load(fname):
    with open(fname) as f:
        records = json.load(f)
    mapping = {record['id']: record for record in records}
    if len(mapping) != records:
        raise ValueError(f'Duplicates record ids exist in {origin_fname}')
    return mapping


def dump_json_records(records, stream):
    first_char = '['
    for r in records:
        stream.write(first_char)
        json.dump(r, stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description='combine results')
    parser.add_argument('base', str, help='[lower precedence]')
    parser.add_argument('update', str, help='[higher precedence]')
    parser.add_argument('out', str, help='output [<base>|<update> allowed]')
    args = parser.parse_args()

    base_map = load(args.base)
    update_map = load(args.update)

    base_map.update(update_map)

    result_list = [base_map[k] for k in sorted(base_map)]
    first_char = '['
    with open(args.out, 'w') as f:
        for record in result_list:
            f.write(first_char)
            json.dump(record, f)
            f.write('\n')
            first_char = ','
        f.write(']\n')

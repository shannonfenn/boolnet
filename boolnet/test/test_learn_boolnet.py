import re
import yaml
import json
import pytest
from glob import glob
from itertools import count
from boolnet.utils import NumpyAwareJSONEncoder
from boolnet.exptools import config_tools
from boolnet.exptools import learn_boolnet


def task_expectation_pairs():
    for folder in glob('boolnet/test/runs/*/'):
        with open(f'{folder}/config.yaml') as f:
            top_config = yaml.load(f)
        tasks, _ = config_tools.generate_tasks(top_config, True)
        with open(f'{folder}/expected.json') as f:
            for i, tsk, exp in zip(count(), tasks, f):
                tsk['id'] = i
                exp = json.loads(exp)
                strip_times(exp)
                yield tsk, exp


def strip_times(record):
    to_remove = [k for k in record if re.fullmatch('.*_time', k)]
    for k in to_remove:
        record.pop(k)


@pytest.mark.parametrize('task,expected', task_expectation_pairs())
def test_run(task, expected):
    print(task['id'], task['learner']['name'])
    result = learn_boolnet.learn_bool_net(task, False)
    strip_times(result)
    result['id'] = task['id']
    # pass through json dump/load to deal with annoying floating point issues
    actual = json.loads(json.dumps(result, cls=NumpyAwareJSONEncoder,
                        separators=(',', ':')))
    # TEMPORARY
    if 'feature_sets' in actual and 'feature_sets' not in expected:
        actual.pop('feature_sets')
    if 'feature_sets' in expected and 'feature_sets' not in actual:
        expected.pop('feature_sets')
    actual.pop('fs_sel_metric', None)
    expected.pop('fs_sel_metric', None)

    print(expected)
    print()
    print(actual)
    # assert False
    assert actual == expected

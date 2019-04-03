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
        print(folder)
        with open(f'{folder}/config.yaml') as f:
            top_config = yaml.load(f)
        tasks, _ = config_tools.generate_tasks(top_config, True)
        with open(f'{folder}/expected.json') as f:
            for i, tsk, exp in zip(count(), tasks, f):
                tsk['id'] = i
                exp = json.loads(exp)
                strip_times(exp)
                yield tsk, exp


def specific_harnesses(name, ids):
    with open(f'boolnet/test/runs/{name}/config.yaml') as f:
        top_config = yaml.load(f)
    tasks, _ = config_tools.generate_tasks(top_config, True)
    with open(f'boolnet/test/runs/{name}/expected.json') as f:
        for i, tsk, exp in zip(count(), tasks, f):
            if not ids or i in ids:
                tsk['id'] = i
                exp = json.loads(exp)
                strip_times(exp)
                yield tsk, exp


def strip_times(record):
    to_remove = [k for k in record if re.fullmatch('.*_time', k)]
    for k in to_remove:
        record.pop(k)


# @pytest.mark.parametrize(
#     'task,expected',
#     specific_harnesses('cpar7', [0,1,2,3,4,5,6,7,10,11,12,13,14]))
# @pytest.mark.parametrize('task,expected', task_expectation_pairs())
@pytest.mark.slow
@pytest.mark.skip(reason='external breaking changes are continuous')
def test_run(task, expected):
    print(task['id'], task['learner']['name'], task['notes_tmt'])
    result = learn_boolnet.learn_bool_net(task)
    strip_times(result)
    result['id'] = task['id']
    # pass through json dump/load to deal with annoying floating point issues
    actual = json.loads(json.dumps(result, cls=NumpyAwareJSONEncoder,
                        separators=(',', ':')))

    print(expected)
    print()
    print(actual)
    assert expected == actual

from copy import deepcopy
from collections import MutableMapping, namedtuple
from good import Schema, Required, In, All, Any, Range, Type, IsDir
from good import message, Allow, Optional
from os import fsync
from os.path import join
import numpy as np
import json

from BoolNet.LearnBoolNet import LEARNERS, OPTIMISERS
from BoolNet.Packing import pack_bool_matrix
from BoolNet.BitError import all_metric_names


class BoolMapping:
    def __init__(self, dataset, Ne):
        self.inputs, self.target = self._validate(dataset, Ne)
        self.Ne = Ne

    @staticmethod
    def _validate(dataset, Ne):
        if dataset.shape[0] == 0:
            raise ValueError('Empty dataset.')
        if dataset.shape[1] != 2:
            raise ValueError(
                'Invalid dataset shape ({}).'.format(dataset.shape))
        inputs = np.asarray(dataset[:, 0].tolist())
        target = np.asarray(dataset[:, 1].tolist())
        if inputs.shape[0] != target.shape[0] != Ne:
            raise ValueError(
                'Dataset input (), target () and Ne () do not match.'.format(
                    inputs.shape[0], target.shape[0], Ne))
        return pack_bool_matrix(inputs), pack_bool_matrix(target)

    @property
    def Ni(self):
        return self.inputs.shape[0]

    @property
    def No(self):
        return self.target.shape[0]

    def toDict(self):
        return {'inputs': self.inputs, 'target': self.target, 'Ne': self.Ne}


Instance = namedtuple('Instance', [
    'training_mapping', 'test_mapping', 'training_indices'])


metrics = list(all_metric_names())


@message('2D array expected')
def is_2d(v):
    if len(v.shape) != 2:
        raise ValueError
    return v


@message('1D integer array expected')
def is_1d(v):
    if len(v.shape) != 1:
        raise ValueError
    return v


@message('Integer array expected')
def is_int_arr(v):
    if not np.issubdtype(v.dtype, np.integer):
        raise ValueError
    return v

SA_schema = Schema({
    'name':           'SA',
    'metric':         In(metrics),
    'num_temps':      All(int, Range(min=1)),
    'init_temp':      Range(min=0.0),
    'temp_rate':      Range(min=0.0, max=1.0),
    'steps_per_temp': All(int, Range(min=1))
    }, default_keys=Required)

SA_VN_schema = Schema({
    'name':             'SA-VN',
    'metric':           In(metrics),
    'num_temps':        All(int, Range(min=1)),
    'init_temp':        Range(min=0.0),
    'temp_rate':        Range(min=0.0, max=1.0),
    'steps_per_temp':   All(int, Range(min=1)),
    'init_move_count':  All(int, Range(min=1))
    }, default_keys=Required)

LAHC_schema = Schema({
    'name':             'LAHC',
    'metric':           In(metrics),
    'cost_list_length': All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1))
    }, default_keys=Required)

LAHC_VN_schema = Schema({
    'name':             'LAHC-VN',
    'metric':           In(metrics),
    'cost_list_length': All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1)),
    'init_move_count':  All(int, Range(min=1))
    }, default_keys=Required)

TS_schema = Schema({
    'name':             'TS',
    'metric':           In(metrics),
    'tabu_period':      All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1))
    }, default_keys=Required)

optimiser_name_schema = Schema({
    'name':     In(OPTIMISERS.keys())
    }, extra_keys=Allow)

optimiser_schema = Schema(
    All(Any(SA_schema, SA_VN_schema, LAHC_schema, LAHC_VN_schema, TS_schema),
        optimiser_name_schema))

sampling_schema = Schema({
    'method':                   In({'given', 'generated'}),
    'Ns':                       All(int, Range(min=1)),
    'Ne':                       All(int, Range(min=1)),
    Optional('indices'):        [All(int, Range(min=0))],
    Optional('file_suffix'):    str
    }, default_keys=Required)

network_schema_given = Schema({
    'method':       'given',
    'node_funcs':   'NAND',
    'file':         All(str, lambda v: v.endswith('.json')),
    'index':        All(int, Range(min=0)),
    }, default_keys=Required)

network_schema_generated = Schema({
    'method':       'generated',
    'Ng':           All(int, Range(min=1)),
    'node_funcs':   In(['NAND', 'NOR', 'random'])
    }, default_keys=Required)

network_schema = Schema(Any(network_schema_given,
                            network_schema_generated))

config_schema = Schema({
    'name':                     str,
    'dataset_dir':              IsDir(),
    'dataset':                  All(str, lambda v: v.endswith('.json')),
    'network':                  network_schema,
    'logging':                  In(['none', 'warning', 'info', 'debug']),
    'learner':                  In(LEARNERS.keys()),
    'optimiser':                optimiser_schema,
    'sampling':                 sampling_schema,
    'configuration_number':     All(int, Range(min=0)),
    'training_set_number':      All(int, Range(min=0)),
    'inter_file_base':          str,
    'training_set':             Type(BoolMapping),
    'test_set':                 Type(BoolMapping),
    'training_indices':         All(Type(np.ndarray), is_1d, is_int_arr),
    Optional('initial_gates'):  All(Type(np.ndarray), is_2d, is_int_arr),
    }, default_keys=Required)


class ExperimentJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, BoolMapping):
            return obj.toDict()
        return json.JSONEncoder.default(self, obj)


def dump_configurations(configurations, stream):
    first = True
    for conf in configurations:
        stream.write('[' if first else ', ')
        json.dump(conf[0], stream, cls=ExperimentJSONEncoder)
    stream.write(']')


def dump_results_partial(results, stream, first):
    if not first:
        stream.write(',')
    json.dump(results, stream, cls=ExperimentJSONEncoder)
    stream.write('\n')
    # ensure data is written to disk immediately
    stream.flush()
    fsync(stream.fileno())


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def partition_and_pack_examples(examples, training_indices):
    ''' Parititions the given function into training and test sets,
        based on the given training indices.'''
    # Parameters
    N = examples.shape[0]
    Ns, Ne = training_indices.shape
    # Generate the test indices array, each row should contain all
    # indices not in the equivalent row of training_indices
    test_indices = np.zeros(shape=(Ns, N - Ne), dtype=int)
    for s in range(Ns):
        test_indices[s] = np.setdiff1d(np.arange(N), training_indices[s])
    # Using numpy's advanced indexing we can get the sets
    training_sets = examples[training_indices]
    test_sets = examples[test_indices]

    instances = []
    for trg, tst, ind in zip(training_sets, test_sets, training_indices):
        instances.append(Instance(
            training_mapping=BoolMapping(trg, Ne),
            test_mapping=BoolMapping(tst, N-Ne),
            training_indices=ind))

    return instances


def load_packed_datasets(settings):
    # load data set from file
    dataset_dir = settings['dataset_dir']
    with open(join(dataset_dir, 'functions', settings['dataset'])) as ds_file:
        ds_settings = json.load(ds_file)
    sample_settings = settings['sampling']

    # load function
    function = np.array(ds_settings['function'])
    # load samples from file
    if sample_settings['method'] == 'given':
        # prepate filename
        if 'file_suffix' in sample_settings:
            file_end = '{}_{}_{}{}.json'.format(
                ds_settings['Ni'], sample_settings['Ns'],
                sample_settings['Ne'], sample_settings['file_suffix'])
        else:
            file_end = '{}_{}_{}.json'.format(
                ds_settings['Ni'], sample_settings['Ns'], sample_settings['Ne'])

        sample_filename = join(dataset_dir, 'samples', file_end)
        # open and load file
        with open(sample_filename) as sample_file:
            training_indices = np.array(json.load(sample_file))
    # generate samples
    elif sample_settings['method'] == 'generated':
        Ns = sample_settings['Ns']
        Ne = sample_settings['Ne']
        # generate
        training_indices = np.random.randint(len(function), size=(Ns, Ne))
    else:
        raise ValueError('Invalid sampling method {}'.format(
                         sample_settings['method']))
    # partition the sets based on loaded indices
    return partition_and_pack_examples(function, training_indices)


def handle_initial_network(settings):
    net_method = settings['network']['method']
    if net_method == 'given':
        dataset_dir = settings['dataset_dir']
        filename = settings['network']['file']
        index = settings['network']['index']
        with open(join(dataset_dir, filename)) as f:
            gates = np.array(json.load(f)[index], dtype=np.uint32)
            settings['initial_gates'] = gates


def generate_configurations(settings, evaluator_class):
    # CAUTION: Will modify the settings parameter!
    # the configurations approach involves essentially having
    # a new settings dict for each configuration and updating
    # it with values in each dict in the configurations list

    variable_sets = settings['configurations']
    # no need to keep this sub-dict around
    settings.pop('configurations')
    # Build up the task list
    tasks = []
    for conf_no, variables in enumerate(variable_sets):
        # keep each configuration isolated
        config_settings = deepcopy(settings)
        # update the settings dict with the values for this configuration
        update_nested(config_settings, variables)
        # record the config number for debuggin
        config_settings['configuration_number'] = conf_no
        # load initial network from file if required
        handle_initial_network(config_settings)
        # load the data for this configuration
        instances = load_packed_datasets(config_settings)
        if 'indices' in config_settings['sampling']:
            config_indices = config_settings['sampling']['indices']
            if any(i >= len(instances) for i in config_indices):
                raise ValueError('\"sampling\" -> indices has elements larger than Ns')
        else:
            config_indices = range(len(instances))
        # for each training set
        for i in config_indices:
            instance = instances[i]
            iteration_settings = deepcopy(config_settings)
            iteration_settings['training_indices'] = instance.training_indices
            iteration_settings['training_set'] = instance.training_mapping
            iteration_settings['test_set'] = instance.test_mapping
            iteration_settings['training_set_number'] = i
            iteration_settings['inter_file_base'] += '{}_{}_'.format(conf_no, i)

            config_schema(iteration_settings)

            # dump the iteration settings out
            tasks.append((iteration_settings, evaluator_class))
    return tasks

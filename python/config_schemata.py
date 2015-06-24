import numpy as np
from good import Schema, Required, In, All, Any, Range, Type, IsDir
from good import message, Allow, Optional

from boolmapping import BoolMapping
from BoolNet.LearnBoolNet import LEARNERS, OPTIMISERS
from BoolNet.BitError import all_metric_names

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

from voluptuous import (
    Schema, In, All, Any, Range, IsDir, ALLOW_EXTRA,
    message, Optional, Exclusive, Length, Invalid)
import boolnet.bintools.functions as fn


def conditionally_required(trigger_key, trigger_val, required_key):
    ''' if trigger_key = trigger_val then required_key must be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] == trigger_val)
        if required and required_key not in d:
            raise Invalid('\'{}\' = \'{}\' requires \'{}\'.'.format(
                trigger_key, trigger_val, required_key))
        # Return the dictionary
        return d
    return validator


def conditionally_forbidden(trigger_key, trigger_val, required_key):
    ''' if trigger_key = trigger_val then required_key must NOT be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] == trigger_val)
        if required and required_key in d:
            raise Invalid('\'{}\' = \'{}\' forbids \'{}\'.'.format(
                trigger_key, trigger_val, required_key))
        # Return the dictionary
        return d
    return validator


@message('Must be a permutation.')
def permutation(l):
    assert (min(l) == 0 and max(l) == len(l) - 1 and len(set(l)) == len(l))
    return l


guiding_functions = fn.scalar_function_names()


data_schema = Any(
    # generated from operator
    Schema({
        'type':                     'generated',
        'operator':                 str,
        'bits':                     All(int, Range(min=1)),
        Optional('out_width'):      All(int, Range(min=1)),
        Optional('window_size'):    All(int, Range(min=1))
        },
        required=True),
    # read from file
    Schema({
        'type':             'file',
        'filename':         str,
        Optional('dir'):    IsDir(),
        },
        required=True)
    )

sampling_schema = Any(
    # randomly generated
    Schema({
        'type': 'generated',
        'Ns':   All(int, Range(min=1)),
        'Ne':   All(int, Range(min=1)),
        'seed': All(int, Range(min=0))
        },
        required=True),
    # read from file
    Schema({
        'type':             'file',
        'filename':         str,
        Optional('dir'):    IsDir(),
        # allow for now, but don't force
        Optional('seed'):   Any(None, All(int, Range(min=0))),
        },
        required=True),
    # given in config file
    Schema({
        'type':             'given',
        'indices':          [[All(int, Range(min=0))]],
        # allow for now, but don't force
        Optional('seed'):   Any(None, All(int, Range(min=0))),
        },
        required=True)
    )


network_schema = All(
    Schema({
        'method':       'generated',
        'Ng':           All(int, Range(min=1)),
        'node_funcs':   [All(int, Range(min=0, max=15))]
        },
        required=True),
    # Schema({
    #     'method':   'given',
    #     'gates':    [All([All(int, Range(min=0))], Length(min=3, max=3))],
    #     })
    )


optimiser_schema = Any(
    # Simulated Annealing
    Schema({
        'name':                     'SA',
        'num_temps':                All(int, Range(min=1)),
        'init_temp':                Range(min=0.0),
        'temp_rate':                Range(min=0.0, max=1.0),
        'steps_per_temp':           All(int, Range(min=1)),
        'guiding_function':         In(guiding_functions),
        Optional('max_restarts'):   All(int, Range(min=0))
        },
        required=True),
    # Hill Climbing
    Schema({
        'name':                     'HC',
        'max_iterations':           All(int, Range(min=1)),
        'guiding_function':         In(guiding_functions),
        Optional('max_restarts'):   All(int, Range(min=0))
        },
        required=True),
    # Late-Acceptance Hill Climbing
    Schema({
        'name':                     'LAHC',
        'cost_list_length':         All(int, Range(min=1)),
        'max_iterations':           All(int, Range(min=1)),
        'guiding_function':         In(guiding_functions),
        Optional('max_restarts'):   All(int, Range(min=0))
        },
        required=True)
    )


target_order_schema = Any('auto', 'msb', 'lsb', All(list, permutation))


learner_schema = Schema(
    All(
        Schema({
            'name':         Any('basic', 'stratified'),
            'network':      network_schema,
            'optimiser':    optimiser_schema,
            'target_order': target_order_schema,
            'seed':         Any(None, All(int, Range(min=0))),
            Optional('minfs_selection_method'): str,
            Optional('minfs_masking'):          bool
            },
            required=True),
        # if target_order = auto then minfs_selection_method must be set
        conditionally_required(
            'target_order', 'auto', 'minfs_selection_method'),
        # if minfs_masking = True then minfs_selection_method must be set
        conditionally_required(
            'minfs_masking', True, 'minfs_selection_method'),
        # if name = basic then minfs_masking is not allowed
        conditionally_forbidden(
            'name', 'basic', 'minfs_masking')
        )
    )


log_keys_schema = Schema(
    [All(Schema([], extra=ALLOW_EXTRA), Length(min=3, max=3))]
    )


instance_schema = Schema({
    'data':     data_schema,
    'learner':  learner_schema,
    'sampling': sampling_schema,
    'log_keys': log_keys_schema,
    Optional('verbose_errors'):             bool,
    Optional('verbose_timing'):             bool,
    Optional('record_final_net'):           bool,
    Optional('record_intermediate_nets'):   bool,
    },
    required=True)


# ########## Schemata for base configs ########## #


seeding_schema = Schema({
    # may be 'shared', 'unique' or any non-negative integer
    'sampling': Any('shared', 'unique', All(int, Range(min=0))),
    'learner':  Any('shared', 'unique', All(int, Range(min=0)))
    })


list_msg = '\'list\' must be a sequence of mappings.'
prod_msg = '\'product\' must be a length 2 sequence of sequences of mappings.'
experiment_schema = Schema({
    'name':                     str,
    'seeding':                  seeding_schema,
    # Must be any dict
    'base_config':              Schema({}, extra=ALLOW_EXTRA),
    # only one of 'list_config' or 'product_config' are allowed
    # Must be a list of dicts
    Exclusive('list', 'config', msg=list_msg):    All(
        Schema([{}], extra=ALLOW_EXTRA), Length(min=1)),
    # Must be a length 2 list of lists of dicts
    Exclusive('product', 'config', msg=prod_msg): All([
        All(Schema([{}], extra=ALLOW_EXTRA), Length(min=1))],
                                    Length(min=2, max=2)),
    # optional level of debug logging
    Optional('debug_level'):        In(['none', 'warning', 'info', 'debug']),
    },
    required=True)

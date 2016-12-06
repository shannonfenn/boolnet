# import yaml
# import json
# import os
# import tempfile
# from pytest import yield_fixture
# from boolnet.exptools.config_tools import generate_configurations
# from boolnet.learn_boolnet import learn_bool_net


# TEST_DIR = 'boolnet/test/runs/'


# # #################### Global fixtures #################### #
# @yield_fixture(params=['basic', 'stratified'])
# def harness(request):
#     # load experiment file
#     print(os.getcwd())
#     with open(TEST_DIR + 'settings-{}.yaml'.format(request.param)) as f:
#         settings = yaml.load(f, Loader=yaml.CSafeLoader)

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         settings['inter_file_base'] = os.path.join(tmpdirname, 'inter_')
#         settings['data']['dir'] = TEST_DIR + 'data/'

#         with open(TEST_DIR + 'expected-{}.json'.format(request.param)) as f:
#             expected = json.load(f)

#         yield settings, expected


# def test(harness):
#     settings, expected = harness
#     configurations = generate_configurations(settings)
#     # Run the actual learning as a parallel process
#     # runs the given configurations
#     actual = map(learn_bool_net, configurations)

#     for exp, act in zip(expected, actual):
#         # assert exp == act
#         for k in exp:
#             print(k)
#             assert exp[k] == act[k]

#     # assert expected == actual

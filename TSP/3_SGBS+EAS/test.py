
"""
The MIT License

Copyright (c) 2022 SGBS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

USE_CUDA = True
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# parameters

#########################
# Parameters - Base
#########################

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

#########################
# Parameters - EAS/SGBS
#########################

run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../1_pre_trained_model/20220226_114432_train__tsp_n100__3000epoch',  # directory path of pre-trained model and log files saved.
        'epoch': 1900,  # epoch version of pre-trained model to laod.
    },
    'test_data_load': {
        'enable': True,
        'filename': '../0_test_data_set/tsp100_test_seed1234.pkl',
        'index_begin': 0,
    },

    'num_episodes': 10000,
    'solution_max_length': 100,  # for buffer length storing solution
    'num_eas_sgbs_loop': 6,

    # EAS Params
    'lr': 0.00815,
    'lambda': 0.006,
    'eas_num_iter': 1,
    'eas_batch_size': 200,

    # SGBS Params
    'beam_width': 10,
    'rollout_per_node': 10-1,
    'sgbs_batch_size': 500,
}




#########################
# Parameters - Log
#########################

logger_params = {
    'log_file': {
        'desc': 'tsp_sgbs_eas',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

import logging
import argparse
from datetime import datetime
import pytz
from utils.utils import create_logger

from TSPTester import TSPTester as Tester


def main():
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                   model_params=model_params,
                   run_params=run_params)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def parse_args(arg_str = None):
    global CUDA_DEVICE_NUM
    global run_params
    global logger_params

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--eas_batch_size", type=int)
    parser.add_argument("--sgbs_batch_size", type=int)
    

    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=arg_str.split())


    if args.ep is not None:
        run_params['num_episodes'] = args.ep

    if args.jump is not None:
        num_episodes = run_params['num_episodes']
        run_params['test_data_load']['index_begin'] += args.jump * num_episodes

    if args.gpu is not None:
        CUDA_DEVICE_NUM = args.gpu
        run_params['cuda_device_num'] = args.gpu


    if args.eas_batch_size is not None:
        run_params['eas_batch_size'] = args.eas_batch_size
    if args.sgbs_batch_size is not None:
        run_params['sgbs_batch_size'] = args.sgbs_batch_size
        

    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
        env_params['pomo_size'] = args.problem_size
        run_params['solution_max_length'] = args.problem_size
        run_params['test_data_load']['filename'] = '../0_test_data_set/tsp{}_test_small_seed1235.pkl'.format(args.problem_size)


if __name__ == "__main__":
    parse_args()
    main()


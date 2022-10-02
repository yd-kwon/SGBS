
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
# Parameters

##################################
# Parameters - From Trained Model
##################################

env_params = {
    'stage_cnt': 3,
    'machine_cnt_list': [4, 4, 4],
    'job_cnt': 100,
    'process_time_params': {
        'time_low': 2,
        'time_high': 10,
    },
    'pomo_size': 24  # assuming 4 machines at each stage! 4*3*2*1
}

model_params = {
    'stage_cnt': env_params['stage_cnt'],
    'machine_cnt_list': env_params['machine_cnt_list'],
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 3,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 4,  # must be >= machine_cnt
}

#########################
# Parameters - EAS/SGBS
#########################

run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../1_pre_trained_model/20220319_162336_ffsp_n100__batch_11_13_13_13',  # directory path of pre-trained model and log files saved.
        'epoch': 120,  # epoch version of pre-trained model to laod.
    },
    'test_data_load': {
        'filename': '../0_test_data_set/unrelated_10000_problems_444_job100_2_10.pt',
        'index_begin': 0,
    },

    'num_episodes': 1000,

    'init_rollout_aug_factor': 128,
    'init_rollout_batch_size': 100,

    'aug_factor': 8,  # augmentation
    'num_eas_sgbs_loop': 5,

    # EAS Params
    'lr': 0.001,
    'lambda': 0.01,
    'eas_num_iter': 1,
    'eas_batch_size': 50,

    # SGBS Params
    'beam_width': 5,
    'rollout_per_node': 6-1,
    'sgbs_batch_size': 2500,
    'sgbs_step_max': 350,
}




#########################
# Parameters - Log
#########################

logger_params = {
    'log_file': {
        'desc': 'ffsp100',
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
from FFSPTester import FFSPTester as Tester


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


def parse_args():
    global CUDA_DEVICE_NUM
    global run_params
    global logger_params

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.ep is not None:
        run_params['num_episodes'] = args.ep

    if args.jump is not None:
        num_episodes = run_params['num_episodes']
        run_params['test_data_load']['index_begin'] += args.jump * num_episodes

    if args.gpu is not None:
        CUDA_DEVICE_NUM = args.gpu
        run_params['cuda_device_num'] = args.gpu


if __name__ == "__main__":
    parse_args()
    main()

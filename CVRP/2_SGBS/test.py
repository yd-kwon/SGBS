
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

##########################################################################################
# parameters

env_params = None
model_params = None
tester_params = None
logger_params = None

def reset_parameters():
    global env_params
    global model_params
    global tester_params
    global logger_params

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
    }


    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        
        'model_load': {
            'path': '../1_pre_trained_model/Saved_CVRP100_Model',   # directory path of pre-trained model and log files saved.
            'epoch': 30500,  # epoch version of pre-trained model to laod.
        },
        
        'test_data_load': {
            'filename': '../0_test_data_set/vrp100_test_seed1234.pkl',
            'index_begin': 0,
        },
        
        'test_episodes': 10000,
        'test_batch_size': 625,

        'augmentation_enable': True,

        'mode': 'sgbs',					# greedy, sampling, obs, mcts, sgbs
        'num_starting_points': 4,
        
        'sampling_num': 1200,           # for sampling
        'obs_bw': 1200,                 # beam_wdith of original beam search    
        'mcts_rollout_per_step': 12,    # number of rollout per step of mcts
        'sgbs_beta': 4,            
        'sgbs_gamma_minus1': (4-1) 
    }


    logger_params = {
        'log_file': {
            'desc': 'cvrp_sgbs',
            'filename': 'log.txt'
        }
    }


reset_parameters()


##########################################################################################
# main

# import

import logging
import argparse

from utils.utils import create_logger
from CVRPTester import CVRPTester as Tester


def main():
    create_logger(**logger_params)
    _print_config()

    if tester_params['mode']=='mcts' and tester_params['augmentation_enable']==True:
        print("Error: mcts with instance augmentation is not supported. Use -disable_aug option.")
        return 0    
    
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    result = tester.run()
    return result


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
    

def parse_args(args_str=None):
    global CUDA_DEVICE_NUM
    global logger_params

    reset_parameters()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)    

    parser.add_argument('-disable_aug', action="store_true", default=False)    
    parser.add_argument('--testdata', type=str)
    parser.add_argument('--problem_size', type=int)
    
    parser.add_argument('--mode', type=str)
    parser.add_argument('--beta', type=int)
    parser.add_argument('--gamma', type=int)

    parser.add_argument('--batch', type=int)
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args_str.split())

    if args.gpu is not None:
        CUDA_DEVICE_NUM = args.gpu
        tester_params['cuda_device_num'] = args.gpu

    if args.disable_aug == True:
        tester_params['augmentation_enable'] = False

    if args.testdata is not None:
        tester_params['test_data_load']['filename'] = args.testdata

    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
        env_params['pomo_size'] = args.problem_size
        logger_params['log_file']['desc'] += '_cvrp{}'.format(args.problem_size)    

    if args.mode is not None:
        tester_params['mode'] = args.mode

    if args.beta is not None:
        tester_params['sgbs_beta'] = args.beta 
        logger_params['log_file']['desc'] += '_beta{}'.format(args.beta )    

    if args.gamma is not None:
        tester_params['sgbs_gamma_minus1'] = args.gamma-1
        logger_params['log_file']['desc'] += '_gamma{}'.format(args.gamma)    

    if args.batch is not None:
        tester_params['test_batch_size'] = args.batch
        logger_params['log_file']['desc'] += '_batch{}'.format(args.batch)

    if args.ep is not None:
        tester_params['test_episodes'] = args.ep
        logger_params['log_file']['desc'] = logger_params['log_file']['desc'] + '_ep{}'.format(args.ep)

    if args.jump is not None:
        num_episodes = tester_params['test_episodes']
        tester_params['test_data_load']['index_begin'] += args.jump * num_episodes
        logger_params['log_file']['desc'] += '_jump{}'.format(args.jump)
        

if __name__ == "__main__":
    parse_args()
    main()
 
    

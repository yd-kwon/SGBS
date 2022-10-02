
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

import torch
import numpy as np

import os
from logging import getLogger
import pickle
import copy

from E_CVRPEnv import E_CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Augmentation
        self.aug_factor = 8 if self.tester_params['augmentation_enable'] else 1

        # utility
        self.time_estimator = TimeEstimator()


    def run(self):
        self.time_estimator.reset()

        # Load problems from problem file
        filename = self.tester_params['test_data_load']['filename']
        index_begin = self.tester_params['test_data_load']['index_begin']
        test_num_episode = self.tester_params['test_episodes']
        self.env.use_pkl_saved_problems(filename, test_num_episode, index_begin)

        # set test method
        test_method_list = {
            'greedy':   self._test_one_batch_greedy,
            'sampling': self._test_one_batch_sampling,
            'obs':      self._test_one_batch_original_beam_search,
            'mcts':     self._test_one_batch_mcts,
            'sgbs':     self._test_one_batch_simulation_guided_beam_search, 
        }

        test_method = test_method_list[self.tester_params['mode']]

        # prepare
        result_arr = torch.zeros(test_num_episode)

        # run
        with torch.no_grad():
            episode = 0
            
            while episode < test_num_episode:
            
                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)            
                batch_score = test_method(episode, batch_size)

                result_arr[episode:episode+batch_size] = batch_score            
                episode += batch_size
            
                # Logs
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info("{:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, score_mean:{:.6f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, 
                    batch_score.mean().item(), result_arr[:episode].mean().item()))
                                            
        # Save Result
        result_to_save = {
            'index_begin': index_begin,
            'num_episode': test_num_episode,
            'result_arr': result_arr.cpu().numpy(),
        }        
        with open('{}/result.pkl'.format(self.result_folder), 'wb') as f:
            pickle.dump(result_to_save, f)                

        # Done
        self.logger.info(" *** Done *** ")
        self.logger.info(" Final Score: {}".format(result_arr.mean().item()))

        return result_arr.mean().item()
        

    def _get_pomo_starting_points(self, model, env, num_starting_points):
        
        # Ready
        ###############################################
        model.eval()
        env.modify_pomo_size(self.env_params['pomo_size'])
        env.reset()

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        
        # starting points
        ###############################################
        sorted_index = reward.sort(dim=1, descending=True).indices
        selected_index = sorted_index[:, :num_starting_points]
        selected_index = selected_index + 1     # depot is 0, and node index starts from 1
        # shape: (batch, num_starting_points)
        
        return selected_index    



    def _test_one_batch_greedy(self, episode, batch_size):

        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, batch_size, self.aug_factor)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward = aug_reward.max(dim=2).values  # get best results from pomo
        # shape: (augmentation, batch)

        max_aug_pomo_reward = max_pomo_reward.max(dim=0).values  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value

        return aug_score



    def _test_one_batch_sampling(self, episode, batch_size):
        num_sampling = self.tester_params['sampling_num']
        aug_batch_size = self.aug_factor * batch_size
        
        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, batch_size, self.aug_factor)

        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)


        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, self.tester_params['num_starting_points'])
        num_repeat = ( num_sampling // starting_points.size(1) ) + 1
        pomo_starting_points = starting_points.repeat(1, num_repeat)[:, :num_sampling]


        # Sampling 
        ###############################################
        self.env.modify_pomo_size(num_sampling)
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)
        
        # the second step, pomo starting points            
        state, _, done = self.env.step(pomo_starting_points)
        
        while not done:
            selected, _ = self.model(state, 'softmax')
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward = aug_reward.max(dim=2).values  # get best results from sampling
        # shape: (augmentation, batch)

        max_aug_pomo_reward = max_pomo_reward.max(dim=0).values  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value

        return aug_score

        
    def _test_one_batch_original_beam_search(self, episode, batch_size):
        beam_width = self.tester_params['obs_bw']        
        aug_batch_size = self.aug_factor * batch_size
    
        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, batch_size, self.aug_factor)
        
        reset_state, _, __ = self.env.reset()
        self.model.pre_forward(reset_state)

        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, self.tester_params['num_starting_points'])


        # Beam Search
        ###############################################
        # reset
        traj_log_prob_sum = torch.zeros(size=(aug_batch_size, starting_points.size(1)))
        self.env.modify_pomo_size(starting_points.size(1))
        self.env.reset()
        
        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points           
        state, _, done = self.env.step(starting_points)

        # LOOP
        while not done:
            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)                
            # shape: (aug_batch, beam_width, problem+1)

            traj_log_prob_sum_exp = traj_log_prob_sum[:, :, None] + probs.log()
            # shape: (aug_batch, beam, problem+1)

            ordered_prob, ordered_i = traj_log_prob_sum_exp.reshape(aug_batch_size, -1).sort(dim=1, descending=True)
            
            ordered_i_selected = ordered_i[:, :beam_width]
            # shape: (aug*batch, beam)
            
            beam_selected = ordered_i_selected // probs.size(2)
            # shape: (aug*batch, beam)
            
            action_selected = ordered_i_selected % probs.size(2)            
            # shape: (aug*batch, beam)
                            
            # traj_log_prob_sum
            ###############################################
            traj_log_prob_sum = ordered_prob[:, :beam_width]

            # BS Step
            ###############################################
            self.env.reset_by_gathering_rollout_env(self.env, gathering_index=beam_selected)
            state, reward, done = self.env.step(action_selected)

    
        # Return
        ###############################################
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
    
        max_pomo_reward = aug_reward.max(dim=2).values  # get best results from beam search
        # shape: (augmentation, batch)
    
        max_aug_pomo_reward = max_pomo_reward.max(dim=0).values  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value
    
        return aug_score


    def _test_one_batch_mcts(self, episode, batch_size):
        reward = torch.zeros(size=(batch_size, ))
        for b in range(batch_size):      
            reward[b] = self._test_one_episode_mcts(episode+b)

        # Return
        ###############################################
        score = -reward
        return score


    def _test_one_episode_mcts(self, episode):
        class Node:
            init_q = -100
            init_qn = 1.0
            cpuct = 2
                
            def __init__(self, node_action, parent_node, child_probs, ninf_mask, env, done):
                self.node_action = node_action
                self.P = child_probs.clone()
                self.ninf_mask = ninf_mask.clone()
                
                self.Q = torch.ones(size=self.P.size())*Node.init_q
                self.N = torch.zeros(size=self.P.size())
                self.zeros = torch.zeros(size=self.P.size())
                self.init_qn = torch.ones(size=self.P.size())*Node.init_qn
                
                self.parent_node = parent_node
                self.child_node = [None]*self.P.size(0)
        
                self.wp = torch.tensor(float('inf'))
                self.bp = torch.tensor(float('-inf'))    

                self.env = copy.deepcopy(env)
                self.done = done
        
        
            def _get_uct(self):
                if self.N.sum() == 0:
                    return self.P
                    
                Qn = (self.Q - self.wp)/(self.bp - self.wp) if self.bp - self.wp > 0 else self.zeros
                Qn = torch.where(self.Q==Node.init_q, self.init_qn, Qn)
                
                U = Node.cpuct * self.P * self.N.sum().sqrt() / ( 0.1 + self.N )
        
                return Qn+U+self.ninf_mask
        
        
            def select_next(self):
                uct = self._get_uct()
                idx = uct.argmax(dim=0)
        
                return self.child_node[idx], idx
        
            def set_child(self, idx, node):
                self.child_node[idx] = node
        
            def set_parent(self, parent_node):    
                self.parent_node = parent_node
                
            def get_parent(self):
                return self.parent_node
        
            def update_q(self, idx, new_q):
                self.Q[idx] = max(self.Q[idx], new_q)
                self.N[idx] += 1
        
                self.bp = max(self.bp, new_q)
                self.wp = min(self.wp, new_q)
        
        
            def select_child_to_move(self):
                Qn_valid = self.Q + self.ninf_mask
                child_idx = Qn_valid.argmax(dim=0)
                    
                return child_idx
             
            def get_child_to_move(self, child_idx):            
                return self.child_node[child_idx]
        
            def get_env_state(self):
                return copy.deepcopy(self.env), self.env.step_state, self.done
                
    
        rollout_per_step = self.tester_params['mcts_rollout_per_step']

        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, 1, self.aug_factor)
        
        reset_state, _, __ = self.env.reset()
        self.model.pre_forward(reset_state)

        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, 1)
        # shape: (aug*batch_size, 1)

        # MCTS
        ###############################################
        # reset
        self.env.modify_pomo_size(starting_points.size(1))
        self.env.reset()
        
        # the first step, depot
        selected = torch.zeros(size=(1, 1), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points           
        state, _, done = self.env.step(starting_points)

        # MCTS Step > 1
        ###############################################

        # LOOP
        next_root = None
        
        while not done:
            node_root = next_root

            if node_root == None:
                probs = self.model.get_expand_prob(state)
                # shape: (aug_batch, 1, problem+1)                

                node_root = Node(0, None, 
                                probs[0, 0], 
                                state.ninf_mask[0, 0], 
                                self.env, done)

            node_root.set_parent(None)


            for mcts_cnt in range(rollout_per_step):
                # selection
                ###############################################
                node_curr = node_root
                node_next, idx_next = node_curr.select_next()

                while node_next is not None:
                    node_curr = node_next
                    node_next, idx_next = node_curr.select_next()

                
                # expansion
                ###############################################
                simulation_env, sim_state, sim_done = node_curr.get_env_state()           
                if sim_done:
                    continue

                sim_state, sim_reward, sim_done = simulation_env.step(idx_next[None, None])                    

                sim_probs = self.model.get_expand_prob(sim_state)
                
                node_exp = Node(idx_next, node_curr, 
                                sim_probs[0, 0], 
                                sim_state.ninf_mask[0, 0], 
                                simulation_env, sim_done)
                node_curr.set_child(idx_next, node_exp)

                # simulation
                ###############################################
                while not sim_done:
                    selected, _ = self.model(sim_state)
                    sim_state, sim_reward, sim_done = simulation_env.step(selected)                    

                new_q = sim_reward[0, 0]

                # backprop
                ###############################################
                node_curr = node_exp
                node_parent = node_curr.get_parent()

                while node_parent is not None:
                    idx_curr = node_curr.node_action
                    node_parent.update_q(idx_curr, new_q)

                    node_curr = node_parent
                    node_parent = node_curr.get_parent()
                                        
            action = node_root.select_child_to_move()      
            next_root = node_root.get_child_to_move(action)
            state, reward, done = self.env.step(action[None, None])
            
        return reward[0, 0]

        
    def _test_one_batch_simulation_guided_beam_search(self, episode, batch_size):
        beam_width = self.tester_params['sgbs_beta']     
        expansion_size_minus1 = self.tester_params['sgbs_gamma_minus1']
        rollout_width = beam_width * expansion_size_minus1
        aug_batch_size = self.aug_factor * batch_size
    
        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, batch_size, self.aug_factor)
        
        reset_state, _, __ = self.env.reset()
        self.model.pre_forward(reset_state)


        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, beam_width)
        

        # Beam Search
        ###############################################
        self.env.modify_pomo_size(beam_width)
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points           
        state, _, done = self.env.step(starting_points)


        # BS Step > 1
        ###############################################

        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.env)
        rollout_env.modify_pomo_size(rollout_width)

        # LOOP
        first_rollout_flag = True
        while not done:

            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :expansion_size_minus1]
                idx_selected = ordered_i[:, :, :expansion_size_minus1]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:expansion_size_minus1+1]
                idx_selected = ordered_i[:, :, 1:expansion_size_minus1+1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.env, repeat=expansion_size_minus1)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                selected, _ = self.model(rollout_state)
                # shape: (aug*batch, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)
            # rollout_reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            rollout_reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, slightly improves performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(self.env)
                rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                # rollout_reward.shape: (aug*batch, rollout_width + beam_width)
                next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            first_rollout_flag = False

            # BS Step
            ###############################################
            sorted_reward, sorted_index = rollout_reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            self.env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            selected = next_nodes.gather(dim=1, index=beam_index)
            # shape: (aug*batch, beam_width)
            state, reward, done = self.env.step(selected)

    
        # Return
        ###############################################
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
    
        max_pomo_reward = aug_reward.max(dim=2).values  # get best results from simulation guided beam search
        # shape: (augmentation, batch)
    
        max_aug_pomo_reward = max_pomo_reward.max(dim=0).values  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value
    
        return aug_score

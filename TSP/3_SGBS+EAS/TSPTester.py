
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
import logging
import time
import copy
import pickle
import os, json
import matplotlib.pyplot as plt

from E_TSPEnv import E_TSPEnv as Env
from E_TSPModel import E_TSPModel as Model

from torch.optim import Adam as Optimizer

from utils.utils import get_result_folder, TimeEstimator, AverageMeter, LogData, util_print_log_array


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 run_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params

        # result folder, logger
        self.logger = logging.getLogger()
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        if self.run_params['use_cuda']:
            cuda_device_num = self.run_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV
        self.env = Env(**self.env_params)
        test_data_load = self.run_params['test_data_load']
        if test_data_load['enable']:
            filename = test_data_load['filename']
            num_problems = self.run_params['num_episodes']
            index_begin = test_data_load['index_begin']
            self.env.use_pkl_saved_problems(filename, num_problems, index_begin)

        # Model
        self.model = Model(**self.model_params)
        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Loading Flags
        self.use_loaded_incumbent = False
        self.use_loaded_eas_params = False
        self.use_loaded_pomo_result = False

        # Storage on CPU
        self.all_incumbent_solution = None
        self.all_incumbent_score = None
        self.all_W1 = None
        self.all_b1 = None
        self.all_W2 = None
        self.all_b2 = None
        self.all_pomo_result = None

        # data in a batch, to be loaded
        self.batch_incumbent_solution = None  # shape: (aug_batch, tour_len)
        self.batch_incumbent_score = None  # shape: (aug_batch,)
        self.batch_W1 = None  # shape: (aug_batch, emb, emb)
        self.batch_b1 = None  # shape: (aug_batch, emb)
        self.batch_W2 = None  # shape: (aug_batch, emb, emb)
        self.batch_b2 = None  # shape: (aug_batch, emb)
        self.batch_pomo_result = None  # shape: (aug_batch, pomo_size)

    def run(self):

        # Storage on CPU & Loading Flags
        self._prep_incumbent_memory_on_cpu()
        self._prep_eas_params_memory_on_cpu()
        self._prep_pomo_result_memory_on_cpu()

        # EAS+SGBS Loop
        self.logger.info("EAS+SGBS Loop Started ")
        start_time = time.time()
        self.time_estimator.reset()

        num_loop = self.run_params['num_eas_sgbs_loop']
        for loop_cnt in range(num_loop):

            # EAS
            self.use_loaded_pomo_result = False
            eas_start_hr = (time.time() - start_time) / 3600.0
            score_curve = self._run_eas(num_iter=self.run_params['eas_num_iter'])
            eas_stop_hr = (time.time() - start_time) / 3600.0

            self.result_log.append('eas_start_time', eas_start_hr)
            self.result_log.append('eas_end_time', eas_stop_hr)
            self.result_log.append('eas_start_score', score_curve[0].item())
            self.result_log.append('eas_end_score', score_curve[-1].item())

            # SGBS
            self.use_loaded_incumbent = True
            self.use_loaded_eas_params = True
            self.use_loaded_pomo_result = True
            sgbs_start_hr = (time.time() - start_time) / 3600.0
            init_score, final_score = self._run_sgbs()
            sgbs_stop_hr = (time.time() - start_time) / 3600.0

            self.result_log.append('sgbs_start_time', sgbs_start_hr)
            self.result_log.append('sgbs_end_time', sgbs_stop_hr)
            self.result_log.append('sgbs_start_score', init_score)
            self.result_log.append('sgbs_end_score', final_score)

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(loop_cnt+1, num_loop)
            self.logger.info("loop {:3d}/{:d}, Elapsed[{}], Remain[{}], score: {:f}".format(
                loop_cnt+1, num_loop, elapsed_time_str, remain_time_str, final_score))

            # Save Result
            with open('{}/result.pkl'.format(self.result_folder), 'wb') as f:
                pickle.dump(self.result_log, f)


        # Done
        self.logger.info(" *** Done *** ")
        util_print_log_array(self.logger, self.result_log)
        self.logger.info(" Final Score: {}".format(final_score))


    def _prep_incumbent_memory_on_cpu(self):

        aug_factor = 8
        num_episode = self.run_params['num_episodes']
        solution_max_length = self.run_params['solution_max_length']

        test_data_load = self.run_params['test_data_load']

        if self.use_loaded_incumbent is False:
            self.all_incumbent_solution = torch.zeros(size=(aug_factor, num_episode, solution_max_length),
                                                      dtype=torch.long, device='cpu')
            self.all_incumbent_score = torch.empty(size=(aug_factor, num_episode,), device='cpu')

    def _prep_eas_params_memory_on_cpu(self):

        aug_factor = 8
        num_episode = self.run_params['num_episodes']
        emb_dim = self.model_params['embedding_dim']  # 128

        test_data_load = self.run_params['test_data_load']

        if self.use_loaded_eas_params is False:
            self.all_eas_W1 = torch.empty(size=(aug_factor, num_episode, emb_dim, emb_dim), device='cpu')
            self.all_eas_b1 = torch.empty(size=(aug_factor, num_episode, emb_dim), device='cpu')
            self.all_eas_W2 = torch.empty(size=(aug_factor, num_episode, emb_dim, emb_dim), device='cpu')
            self.all_eas_b2 = torch.empty(size=(aug_factor, num_episode, emb_dim), device='cpu')

    def _prep_pomo_result_memory_on_cpu(self):

        aug_factor = 8
        num_episode = self.run_params['num_episodes']
        pomo_size = self.env_params['pomo_size']
        self.all_pomo_result = torch.empty(size=(aug_factor, num_episode, pomo_size), device='cpu')


    def _feed_loaded_data_in_batch(self, start_idx, batch_size):

        aug_factor = 8

        # saved incumbent solution
        if self.use_loaded_incumbent:
            incumbent_solution = self.all_incumbent_solution[:, start_idx:start_idx+batch_size]
            incumbent_score = self.all_incumbent_score[:, start_idx:start_idx+batch_size]
            self.batch_incumbent_solution = incumbent_solution.to(self.device).reshape(aug_factor*batch_size, -1)
            self.batch_incumbent_score = incumbent_score.to(self.device).reshape(aug_factor*batch_size)

        # saved eas parameters
        emb_dim = self.model_params['embedding_dim']  # 128
        if self.use_loaded_eas_params:
            W1 = self.all_eas_W1[:, start_idx:start_idx+batch_size]
            b1 = self.all_eas_b1[:, start_idx:start_idx+batch_size]
            W2 = self.all_eas_W2[:, start_idx:start_idx+batch_size]
            b2 = self.all_eas_b2[:, start_idx:start_idx+batch_size]
            self.batch_W1 = W1.to(self.device).reshape(aug_factor*batch_size, emb_dim, emb_dim)
            self.batch_b1 = b1.to(self.device).reshape(aug_factor*batch_size, emb_dim)
            self.batch_W2 = W2.to(self.device).reshape(aug_factor*batch_size, emb_dim, emb_dim)
            self.batch_b2 = b2.to(self.device).reshape(aug_factor*batch_size, emb_dim)

        # saved pomo result (only needed for SGBS, though)
        pomo_size = self.env_params['pomo_size'] # 100
        if self.use_loaded_pomo_result:
            pomo_result = self.all_pomo_result[:, start_idx:start_idx+batch_size]
            self.batch_pomo_result = pomo_result.to(self.device).reshape(aug_factor*batch_size, pomo_size)

    ###########################################################################################################
    ###########################################################################################################
    # EAS
    ###########################################################################################################
    ###########################################################################################################

    def _run_eas(self, num_iter=1):

        result_curve = torch.zeros(size=(num_iter,))

        # Loop
        num_episode = self.run_params['num_episodes']
        episode = 0
        while episode < num_episode:

            remaining = num_episode - episode
            batch_size = min(self.run_params['eas_batch_size'], remaining)

            # EAS
            sum_score_curve = self._eas_one_batch(episode, batch_size, num_iter)
            # shape: (num_iter,)
            result_curve += sum_score_curve

            episode += batch_size

            self.logger.info("\teas batch [{}:{}] score: {:f}".format(
                episode-batch_size, episode, sum_score_curve[-1].item()/batch_size))
            

        # Done
        score_curve = result_curve / num_episode
        return score_curve

    def _eas_one_batch(self, episode, batch_size, num_iter):

        aug_factor = 8
        aug_batch_size = batch_size * aug_factor
        pomo_size = self.env_params['pomo_size']
        sum_score_curve = torch.empty(size=(num_iter,))

        # Ready
        ###############################################
        self.env.load_problems_by_index(episode, batch_size, aug_factor)
        self.env.modify_pomo_size_for_eas(pomo_size)
        reset_state, _, _ = self.env.reset()

        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        # Initial Incumbent
        ###############################################
        self._feed_loaded_data_in_batch(episode, batch_size)

        if self.use_loaded_incumbent:
            incumbent_solution = self.batch_incumbent_solution
            # shape: (aug_batch, solution_max_length)
            incumbent_score = self.batch_incumbent_score
            # shape: (aug_batch,)
        else:
            incumbent_solution, incumbent_score = self._initial_pomo_greedy_rollout()

        # EAS
        ###############################################
        self.model.train()  # Must be in "train mode" for EAS to work
        self.model.decoder.enable_EAS = True

        if self.use_loaded_eas_params:
            self.model.decoder.init_eas_layers_manual(self.batch_W1, self.batch_b1, self.batch_W2, self.batch_b2)
        else:
            self.model.decoder.init_eas_layers_random(aug_batch_size)

        optimizer = Optimizer(self.model.decoder.eas_parameters(), lr=self.run_params['lr'])

        pomo_size_p1 = pomo_size + 1
        self.env.modify_pomo_size_for_eas(pomo_size_p1)

        for iter_i in range(num_iter):
            self.env.reset()
            prob_list = torch.zeros(size=(aug_batch_size, pomo_size_p1, 0))

            # POMO Rollout with Incumbent
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                # Best_Action from incumbent solution
                step_cnt = self.env.selected_count
                best_action = incumbent_solution[:, step_cnt]

                selected, prob = self.model.forward_w_incumbent(state, best_action)
                # shape: (aug_batch, pomo+1)

                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            # Incumbent solution
            ###############################################
            max_reward, max_idx = reward.max(dim=1)  # get best results from pomo + Incumbent
            # shape: (aug_batch,)
            incumbent_score = -max_reward

            gathering_index = max_idx[:, None, None].expand(-1, 1, self.env.selected_count)
            new_incumbent_solution = self.env.selected_node_list.gather(dim=1, index=gathering_index)
            new_incumbent_solution = new_incumbent_solution.squeeze(dim=1)
            # shape: (aug_batch, tour_len)

            solution_max_length = self.run_params['solution_max_length']
            incumbent_solution = torch.zeros(size=(aug_factor*batch_size, solution_max_length), dtype=torch.long)
            incumbent_solution[:, :self.env.selected_count] = new_incumbent_solution

            # Loss: POMO RL
            ###############################################
            pomo_prob_list = prob_list[:, :pomo_size, :]
            # shape: (aug_batch, pomo, tour_len)
            pomo_reward = reward[:, :pomo_size]
            # shape: (aug_batch, pomo)

            advantage = pomo_reward - pomo_reward.mean(dim=1, keepdim=True)
            # shape: (aug_batch, pomo)
            log_prob = pomo_prob_list.log().sum(dim=2)
            # size = (aug_batch, pomo)
            loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD
            # shape: (aug_batch, pomo)
            loss_RL = loss_RL.mean(dim=1)
            # shape: (aug_batch,)

            # Loss: IL
            ###############################################
            imitation_prob_list = prob_list[:, -1, :]
            # shape: (aug_batch, tour_len)
            log_prob = imitation_prob_list.log().sum(dim=1)
            # shape: (aug_batch,)
            loss_IL = -log_prob  # Minus Sign: to increase probability
            # shape: (aug_batch,)

            # Back Propagation
            ###############################################
            optimizer.zero_grad()

            loss = loss_RL + self.run_params['lambda'] * loss_IL
            # shape: (aug_batch,)
            loss.sum().backward()

            optimizer.step()

            # Score Curve
            ###############################################
            augbatch_reward = max_reward.reshape(aug_factor, batch_size)
            # shape: (augmentation, batch)
            max_aug_reward, _ = augbatch_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            sum_score = -max_aug_reward.sum()  # negative sign to make positive value
            sum_score_curve[iter_i] = sum_score

        # Store Result Tensors in CPU_Memory
        ###############################################
        cpu_solution = incumbent_solution.reshape(aug_factor, batch_size, -1).to('cpu')
        self.all_incumbent_solution[:, episode:episode+batch_size] = cpu_solution
        cpu_score = incumbent_score.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_score[:, episode:episode+batch_size] = cpu_score

        emb_dim = self.model_params['embedding_dim']  # 128
        cpu_W1 = self.model.decoder.eas_W1.data.reshape(aug_factor, batch_size, emb_dim, emb_dim).to('cpu')
        self.all_eas_W1[:, episode:episode+batch_size] = cpu_W1
        cpu_b1 = self.model.decoder.eas_b1.data.reshape(aug_factor, batch_size, emb_dim).to('cpu')
        self.all_eas_b1[:, episode:episode+batch_size] = cpu_b1
        cpu_W2 = self.model.decoder.eas_W2.data.reshape(aug_factor, batch_size, emb_dim, emb_dim).to('cpu')
        self.all_eas_W2[:, episode:episode+batch_size] = cpu_W2
        cpu_b2 = self.model.decoder.eas_b2.data.reshape(aug_factor, batch_size, emb_dim).to('cpu')
        self.all_eas_b2[:, episode:episode+batch_size] = cpu_b2

        cpu_pomo_result = pomo_reward.reshape(aug_factor, batch_size, pomo_size).to('cpu')
        self.all_pomo_result[:, episode:episode+batch_size] = cpu_pomo_result

        return sum_score_curve

    def _initial_pomo_greedy_rollout(self):

        self.model.eval()
        self.model.decoder.enable_EAS = False

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Score
        ###############################################
        max_pomo_reward, max_pomo_idx = reward.max(dim=1)  # get best results from pomo
        # shape: (aug_batch,)
        incumbent_score = -max_pomo_reward

        # Solution
        ###############################################
        all_solutions = self.env.selected_node_list
        # shape: (aug_batch, pomo, tour_len)
        tour_len = all_solutions.size(2)
        gathering_index = max_pomo_idx[:, None, None].expand(-1, 1, tour_len)
        best_solution = all_solutions.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (aug_batch, tour_len)

        aug_batch_size = best_solution.size(0)
        solution_max_length = self.run_params['solution_max_length']
        incumbent_solution = torch.zeros(size=(aug_batch_size, solution_max_length), dtype=torch.long)
        incumbent_solution[:, :tour_len] = best_solution

        return incumbent_solution, incumbent_score

    ###########################################################################################################
    ###########################################################################################################
    # SGBS
    ###########################################################################################################
    ###########################################################################################################

    def _run_sgbs(self):

        # result_curve
        score_AM = AverageMeter()
        solution_max_length = self.run_params['solution_max_length']
        result_curve = torch.zeros(size=(solution_max_length,))

        # Loop
        num_episode = self.run_params['num_episodes']
        episode = 0
        while episode < num_episode:

            remaining = num_episode - episode
            batch_size = min(self.run_params['sgbs_batch_size'], remaining)

            # SGBS
            with torch.no_grad():
                final_sum_score, sum_score_curve = self._sgbs_one_batch(episode, batch_size)
                score_AM.update(final_sum_score/batch_size, batch_size)
                result_curve += sum_score_curve

            episode += batch_size

            self.logger.info("\tsgbs batch [{}:{}] score: {:f}".format(
                episode-batch_size, episode, score_AM.avg))

        # Done
        init_score = result_curve[0].item() / num_episode
        final_score = score_AM.avg
        return init_score, final_score

    def _sgbs_one_batch(self, episode, batch_size):

        aug_factor = 8
        aug_batch_size = batch_size * aug_factor
        solution_max_length = self.run_params['solution_max_length']
        sum_score_curve = torch.zeros(size=(solution_max_length,))

        beam_width = self.run_params['beam_width']
        rollout_per_node = self.run_params['rollout_per_node']
        rollout_width = beam_width * rollout_per_node

        # Ready
        ###############################################
        self.env.load_problems_by_index(episode, batch_size, aug_factor)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        self.model.eval()
        self.model.requires_grad_(False)

        self._feed_loaded_data_in_batch(episode, batch_size)
        if self.use_loaded_eas_params:
            self.model.decoder.enable_EAS = True
            self.model.decoder.init_eas_layers_manual(self.batch_W1, self.batch_b1, self.batch_W2, self.batch_b2)

        # BS Step 1
        ###############################################
        bs_env_params = {
            'problem_size': self.env_params['problem_size'],
            'pomo_size': beam_width,
        }
        bs_env = Env(**bs_env_params)
        bs_env.copy_problems(self.env)
        bs_env.reset()

        bs_step = 1
        reward = self.batch_pomo_result

        sorted_reward, sorted_index = reward.sort(dim=1, descending=True)
        beam_reward = sorted_reward[:, :beam_width]
        beam_index = sorted_index[:, :beam_width]
        # shape: (aug*batch, beam_width)

        bs_state, _, bs_done = bs_env.step(beam_index)

        # Summed Scores
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        # shape: (aug*batch,)
        if self.use_loaded_incumbent:
            old_better = self.batch_incumbent_score < -max_pomo_reward
            max_pomo_reward[old_better] = -self.batch_incumbent_score[old_better]
        aug_reward = max_pomo_reward.reshape(aug_factor, batch_size)
        # shape: (aug, batch)
        max_aug_reward, _ = aug_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        sum_score = -max_aug_reward.sum()  # negative sign to make positive value
        sum_score_curve[bs_step-1] = sum_score  # first step

        # BS Step > 1
        ###############################################
        # Prepare Rollout-Env
        rollout_env_params = {
            'problem_size': self.env_params['problem_size'],
            'pomo_size': rollout_width,
        }
        rollout_env = Env(**rollout_env_params)
        rollout_env.copy_problems(self.env)
        rollout_env.reset()

        # LOOP
        first_rollout_flag = True
        while not bs_done:

            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(bs_state)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :rollout_per_node]
                idx_selected = ordered_i[:, :, :rollout_per_node]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:rollout_per_node+1]
                idx_selected = ordered_i[:, :, 1:rollout_per_node+1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, rollout_per_node)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(bs_env, repeat=rollout_per_node)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_factor*batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            state, reward, done = rollout_env.step(next_nodes)
            while not done:
                selected, _ = self.model(state)
                # shape: (aug*batch, rollout_width)
                state, reward, done = rollout_env.step(selected)
            # reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_factor*batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, for slightly better performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(bs_env)
                reward = torch.cat((reward, beam_reward), dim=1)
                # reward.shape: (aug*batch, rollout_width + beam_width)
                next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            first_rollout_flag = False

            # BS Step
            ###############################################
            bs_step += 1

            sorted_reward, sorted_index = reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            bs_env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            bs_selected = next_nodes.gather(dim=1, index=beam_index)
            # shape: (aug*batch, beam_width)
            bs_state, bs_reward, bs_done = bs_env.step(bs_selected)

            # Score Curve
            ###############################################
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from rollout
            # shape: (aug*batch,)
            if self.use_loaded_incumbent:
                old_better = self.batch_incumbent_score < -max_pomo_reward
                max_pomo_reward[old_better] = -self.batch_incumbent_score[old_better]
            aug_reward = max_pomo_reward.reshape(aug_factor, batch_size)
            # shape: (aug, batch)
            max_aug_reward, _ = aug_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            sum_score = -max_aug_reward.sum()  # negative sign to make positive value
            sum_score_curve[bs_step-1] = sum_score

        # Store Result Tensors in CPU_Memory
        ###############################################
        max_reward, max_idx = bs_reward.max(dim=1)  # get best results from beam_width
        # shape: (aug_batch,)
        incumbent_score = -max_reward
        # shape: (aug_batch,)

        gathering_index = max_idx[:, None, None].expand(-1, 1, bs_env.selected_count)
        new_incumbent_solution = bs_env.selected_node_list.gather(dim=1, index=gathering_index).squeeze(dim=1)
        solution_max_length = self.run_params['solution_max_length']
        incumbent_solution = torch.zeros(size=(aug_factor*batch_size, solution_max_length), dtype=torch.long)
        incumbent_solution[:, :bs_env.selected_count] = new_incumbent_solution

        if self.use_loaded_incumbent:
            old_better = self.batch_incumbent_score < incumbent_score
            incumbent_score[old_better] = self.batch_incumbent_score[old_better]
            incumbent_solution[old_better] = self.batch_incumbent_solution[old_better]

        cpu_solution = incumbent_solution.reshape(aug_factor, batch_size, -1).to('cpu')
        self.all_incumbent_solution[:, episode:episode+batch_size] = cpu_solution
        cpu_score = incumbent_score.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_score[:, episode:episode+batch_size] = cpu_score

        return sum_score.item(), sum_score_curve



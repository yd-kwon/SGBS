
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

from E_FFSPEnv import E_FFSPEnv as Env
from E_FFSPModel import E_FFSPModel as Model

from torch.optim import Adam as Optimizer

from utils.utils import get_result_folder, TimeEstimator, AverageMeter, LogData, util_print_log_array


class FFSPTester:
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

        filename = self.run_params['test_data_load']['filename']
        num_problems = self.run_params['num_episodes']
        index_begin = self.run_params['test_data_load']['index_begin']

        data = torch.load(filename)
        problems_INT_list = data['problems_INT_list']
        for stage_idx in range(data['stage_cnt']):
            problems_INT_list[stage_idx] = problems_INT_list[stage_idx][index_begin:index_begin+num_problems]
        for stage_idx in range(data['stage_cnt']):
            problems_INT_list[stage_idx] = problems_INT_list[stage_idx].to(self.device)

        self.ALL_problems_INT_list = problems_INT_list

        # Model
        self.model = Model(**self.model_params)
        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Storage on CPU
        self.all_encoded_row_list = None
        self.all_encoded_col_list = None
        self.all_incumbent_solution = None
        self.all_incumbent_score = None
        self.all_incumbent_pomo_idx = None
        self.all_W1 = None
        self.all_b1 = None
        self.all_W2 = None
        self.all_b2 = None
        self.all_pomo_result = None

        # data in a batch, to be loaded
        self.batch_encoded_row_list = [None]*self.model_params['stage_cnt']  # shape: stage_cnt * (aug_batch, job_cnt, emb)
        self.batch_encoded_col_list = [None]*self.model_params['stage_cnt']   # shape: stage_cnt * (aug_batch, total_machine_cnt, emb)
        self.batch_incumbent_solution = None  # shape: (aug_batch, tour_len)
        self.batch_incumbent_score = None  # shape: (aug_batch,)
        self.batch_incumbent_pomo_idx = None  # shape: (aug_batch,)
        self.batch_W1_list = [None]*self.model_params['stage_cnt']  # shape: stage_cnt * (aug_batch, emb, emb)
        self.batch_b1_list = [None]*self.model_params['stage_cnt']  # shape: stage_cnt * (aug_batch, emb)
        self.batch_W2_list = [None]*self.model_params['stage_cnt']  # shape: stage_cnt * (aug_batch, emb, emb)
        self.batch_b2_list = [None]*self.model_params['stage_cnt']  # shape: stage_cnt * (aug_batch, emb)
        self.batch_pomo_result = None  # shape: (aug_batch, pomo_size)

    def run(self):
        start_time = time.time()

        # Storage on CPU
        self.logger.info("Preparing Data Storage")
        self._prep_cpu_storage()
        self._init_cpu_data()

        # EAS+SGBS Loop
        self.logger.info("EAS+SGBS Loop Starts")
        num_loop = self.run_params['num_eas_sgbs_loop']

        self.time_estimator.reset()
        for loop_cnt in range(num_loop):

            # EAS
            eas_start_hr = (time.time() - start_time) / 3600.0
            score_curve = self._run_eas(num_iter=self.run_params['eas_num_iter'])
            eas_stop_hr = (time.time() - start_time) / 3600.0

            self.result_log.append('eas_start_time', eas_start_hr)
            self.result_log.append('eas_end_time', eas_stop_hr)
            self.result_log.append('eas_start_score', score_curve[0].item())
            self.result_log.append('eas_end_score', score_curve[-1].item())

            # SGBS
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


    def _prep_cpu_storage(self):

        aug_factor = self.run_params['aug_factor']
        num_episode = self.run_params['num_episodes']
        job_cnt = self.env_params['job_cnt']
        machine_cnt_list = self.env_params['machine_cnt_list']
        total_machine_cnt = sum(machine_cnt_list)
        emb_dim = self.model_params['embedding_dim']  # 128
        pomo_size = self.env_params['pomo_size']
        stage_cnt = self.model_params['stage_cnt']

        # encodings
        self.all_encoded_row_list = [None]*stage_cnt
        self.all_encoded_col_list = [None]*stage_cnt
        for stage_idx in range(stage_cnt):
            all_encoded_row = torch.empty(size=(aug_factor, num_episode, job_cnt, emb_dim,), device='cpu')
            all_encoded_col = torch.empty(size=(aug_factor, num_episode, machine_cnt_list[stage_idx], emb_dim,), device='cpu')
            self.all_encoded_row_list[stage_idx] = all_encoded_row
            self.all_encoded_col_list[stage_idx] = all_encoded_col

        # incumbent
        self.all_incumbent_solution = torch.empty(size=(aug_factor, num_episode, total_machine_cnt, job_cnt),
                                                  dtype=torch.long, device='cpu')
        self.all_incumbent_score = torch.empty(size=(aug_factor, num_episode), dtype=torch.long, device='cpu')
        self.all_incumbent_pomo_idx = torch.empty(size=(aug_factor, num_episode), dtype=torch.long, device='cpu')

        # EAS parameters
        self.all_eas_W1_list = [None]*stage_cnt
        self.all_eas_b1_list = [None]*stage_cnt
        self.all_eas_W2_list = [None]*stage_cnt
        self.all_eas_b2_list = [None]*stage_cnt
        init_lim = (1/emb_dim)**(1/2)
        for stage_idx in range(stage_cnt):
            W1= torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((aug_factor, num_episode, emb_dim, emb_dim))
            b1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((aug_factor, num_episode, emb_dim))
            W2 = torch.zeros(size=(aug_factor, num_episode, emb_dim, emb_dim), device='cpu')
            b2 = torch.zeros(size=(aug_factor, num_episode, emb_dim), device='cpu')
            self.all_eas_W1_list[stage_idx] = W1
            self.all_eas_b1_list[stage_idx] = b1
            self.all_eas_W2_list[stage_idx] = W2
            self.all_eas_b2_list[stage_idx] = b2

        # pomo result
        self.all_pomo_result = torch.empty(size=(aug_factor, num_episode, pomo_size), dtype=torch.long, device='cpu')

    def _init_cpu_data(self):

        num_episode = self.run_params['num_episodes']

        # encodings + incumbent
        episode = 0
        while episode < num_episode:

            remaining = num_episode - episode
            batch_size = min(self.run_params['init_rollout_batch_size'], remaining)

            self._init_encodings_and_incumbent_one_batch(episode, batch_size)
            episode += batch_size

        # EAS parameters
        pass  # data is already initialized during storage prep

        # pomo result
        pass  # it will be automatically updated during EAS

    def _init_encodings_and_incumbent_one_batch(self, episode, batch_size):

        init_rollout_aug_factor = self.run_params['init_rollout_aug_factor']
        aug_factor = self.run_params['aug_factor']
        job_cnt = self.env_params['job_cnt']
        machine_cnt_list = self.env_params['machine_cnt_list']
        total_machine_cnt = sum(machine_cnt_list)
        stage_cnt = self.model_params['stage_cnt']
        embedding_dim = self.model_params['embedding_dim']

        # Ready
        ###############################################
        problems_INT_list = []
        for stage_idx in range(stage_cnt):
            problems_INT_list.append(self.ALL_problems_INT_list[stage_idx][episode:episode+batch_size])
            problems_INT_list[stage_idx] = problems_INT_list[stage_idx].repeat(init_rollout_aug_factor, 1, 1)
            # shape: (aug_batch_size, job_cnt, machine_cnt)

        self.env.load_problems_manual(problems_INT_list)
        self.env.set_pomo_idx(self.env.POMO_IDX)
        reset_state, _, _ = self.env.reset()

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.enable_EAS(False)
        self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Select Augmentation
        ###############################################
        max_pomo_reward, max_pomo_idx = reward.max(dim=1)  # get best results from pomo
        # shape: (init_rollout_aug_batch,)

        init_rollout_augs_reward = max_pomo_reward.reshape(init_rollout_aug_factor, batch_size)
        # shape: (init_rollout_aug, batch)
        sorted_reward, sorted_index = init_rollout_augs_reward.sort(dim=0, descending=True)
        selected_aug_reward = sorted_reward[:aug_factor, :]
        selected_aug_index = sorted_index[:aug_factor, :]
        # shape: (aug, batch)
        offset = torch.arange(batch_size)[None, :]
        aug_select_idx = (selected_aug_index*batch_size + offset).reshape(aug_factor*batch_size,)
        pomo_select_idx = max_pomo_idx[aug_select_idx]

        # Score
        ###############################################
        incumbent_score = -selected_aug_reward.reshape(aug_factor*batch_size,)
        # shape: (aug_batch,)

        # Solution
        ###############################################
        incumbent_solution = self.env.schedule[aug_select_idx, pomo_select_idx, :, :job_cnt]
        # shape: (aug_batch, total_machine, job)

        # Store Result Tensors into CPU_Memory
        ###############################################
        for stage_idx in range(stage_cnt):
            encoded_row = self.model.stage_models[stage_idx].encoded_row[aug_select_idx, :, :]
            encoded_col = self.model.stage_models[stage_idx].encoded_col[aug_select_idx, :, :]
            # shape: (aug_batch, job_cnt/machine_cnt, embedding)
            cpu_encoded_row = encoded_row.reshape(aug_factor, batch_size, job_cnt, embedding_dim).to('cpu')
            cpu_encoded_col = encoded_col.reshape(aug_factor, batch_size, machine_cnt_list[stage_idx], embedding_dim).to('cpu')
            self.all_encoded_row_list[stage_idx][:, episode:episode+batch_size] = cpu_encoded_row
            self.all_encoded_col_list[stage_idx][:, episode:episode+batch_size] = cpu_encoded_col

        cpu_solution = incumbent_solution.reshape(aug_factor, batch_size, total_machine_cnt, job_cnt).to('cpu')
        self.all_incumbent_solution[:, episode:episode+batch_size] = cpu_solution
        cpu_score = incumbent_score.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_score[:, episode:episode+batch_size] = cpu_score
        cpu_pomo_idx = pomo_select_idx.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_pomo_idx[:, episode:episode+batch_size] = cpu_pomo_idx

    def _feed_loaded_data_in_batch(self, start_idx, batch_size, load_pomo_result=False):

        aug_factor = self.run_params['aug_factor']
        num_episode = self.run_params['num_episodes']
        job_cnt = self.env_params['job_cnt']
        machine_cnt_list = self.env_params['machine_cnt_list']
        total_machine_cnt = sum(machine_cnt_list)
        emb_dim = self.model_params['embedding_dim']  # 128
        pomo_size = self.env_params['pomo_size']
        stage_cnt = self.model_params['stage_cnt']

        # encodings
        for stage_idx in range(stage_cnt):
            encoded_row = self.all_encoded_row_list[stage_idx][:, start_idx:start_idx+batch_size]
            encoded_col = self.all_encoded_col_list[stage_idx][:, start_idx:start_idx+batch_size]
            self.batch_encoded_row_list[stage_idx] = encoded_row.to(self.device).reshape(aug_factor*batch_size, -1, emb_dim)
            self.batch_encoded_col_list[stage_idx] = encoded_col.to(self.device).reshape(aug_factor*batch_size, -1, emb_dim)

        # incumbent
        incumbent_solution = self.all_incumbent_solution[:, start_idx:start_idx+batch_size]
        incumbent_score = self.all_incumbent_score[:, start_idx:start_idx+batch_size]
        incumbent_pomo_idx = self.all_incumbent_pomo_idx[:, start_idx:start_idx+batch_size]
        self.batch_incumbent_solution = incumbent_solution.to(self.device).reshape(aug_factor*batch_size, total_machine_cnt, job_cnt)
        self.batch_incumbent_score = incumbent_score.to(self.device).reshape(aug_factor*batch_size)
        self.batch_incumbent_pomo_idx = incumbent_pomo_idx.to(self.device).reshape(aug_factor*batch_size)

        # EAS parameters
        for stage_idx in range(stage_cnt):
            W1 = self.all_eas_W1_list[stage_idx][:, start_idx:start_idx+batch_size]
            b1 = self.all_eas_b1_list[stage_idx][:, start_idx:start_idx+batch_size]
            W2 = self.all_eas_W2_list[stage_idx][:, start_idx:start_idx+batch_size]
            b2 = self.all_eas_b2_list[stage_idx][:, start_idx:start_idx+batch_size]
            self.batch_W1_list[stage_idx] = W1.to(self.device).reshape(aug_factor*batch_size, emb_dim, emb_dim)
            self.batch_b1_list[stage_idx] = b1.to(self.device).reshape(aug_factor*batch_size, emb_dim)
            self.batch_W2_list[stage_idx] = W2.to(self.device).reshape(aug_factor*batch_size, emb_dim, emb_dim)
            self.batch_b2_list[stage_idx] = b2.to(self.device).reshape(aug_factor*batch_size, emb_dim)

        # pomo result (only needed for SGBS first step)
        if load_pomo_result:
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

        # Done
        score_curve = result_curve / num_episode
        return score_curve

    def _eas_one_batch(self, episode, batch_size, num_iter):

        aug_factor = self.run_params['aug_factor']
        job_cnt = self.env_params['job_cnt']
        machine_cnt_list = self.env_params['machine_cnt_list']
        total_machine_cnt = sum(machine_cnt_list)
        pomo_size = self.env_params['pomo_size']
        stage_cnt = self.model_params['stage_cnt']
        emb_dim = self.model_params['embedding_dim']

        sum_score_curve = torch.empty(size=(num_iter,))

        # Ready
        ###############################################
        self._feed_loaded_data_in_batch(episode, batch_size)

        problems_INT_list = []
        for stage_idx in range(self.env.stage_cnt):
            problems_INT_list.append(self.ALL_problems_INT_list[stage_idx][episode:episode+batch_size])
            problems_INT_list[stage_idx] = problems_INT_list[stage_idx].repeat(aug_factor, 1, 1)
            # shape: (aug_batch_size, job_cnt, machine_cnt)
        self.env.load_problems_manual(problems_INT_list)

        self.model.train()  # Must be in "train mode" for EAS to work
        self.model.requires_grad_(False)
        self.model.enable_EAS(True)
        self.model.pre_forward_w_saved_encodings(self.batch_encoded_row_list, self.batch_encoded_col_list)
        self.model.init_eas_layers_manual(self.batch_W1_list, self.batch_b1_list, self.batch_W2_list, self.batch_b2_list)

        # EAS
        ###############################################
        incumbent_solution = self.batch_incumbent_solution
        # shape: (aug_batch, total_machine_cnt, job_cnt)
        incumbent_pomo_idx = self.batch_incumbent_pomo_idx
        # shape: (aug_batch,)

        optimizer = Optimizer(self.model.eas_parameters(), lr=self.run_params['lr'])

        pomo_size_p1 = pomo_size + 1
        self.env.modify_pomo_size(pomo_size_p1)

        for iter_i in range(num_iter):
            pomo_idx = self.env.POMO_IDX.clone()
            pomo_idx[:, -1] = incumbent_pomo_idx
            self.env.set_pomo_idx(pomo_idx)
            self.env.reset()
            prob_list = torch.zeros(size=(aug_factor*batch_size, pomo_size_p1, 0))

            # POMO Rollout with Incumbent
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                best_action = self.env.extract_action_from_schedule(incumbent_solution)

                selected, prob = self.model.forward_w_incumbent(state, best_action)
                # shape: (aug_batch, pomo+1)

                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Incumbents
            ###############################################
            reward_w_slope = reward.float() + torch.arange(pomo_size_p1)[None, :].float()/(100*pomo_size_p1)  # add slope to select incumbent when tie
            max_pomo_idx = reward_w_slope.argmax(dim=1)  # get best results from pomo + Incumbent
            # shape: (aug_batch,)
            incumbent_score = -reward[torch.arange(aug_factor*batch_size), max_pomo_idx]

            all_schedules = self.env.schedule[:, :, :, :job_cnt]
            # shape: (aug_batch, pomo+1, total_machine, job)
            gathering_index = max_pomo_idx[:, None, None, None].expand(-1, 1, total_machine_cnt, job_cnt)
            incumbent_solution = all_schedules.gather(dim=1, index=gathering_index).squeeze(dim=1)
            # shape: (aug_batch, total_machine, job)

            inc_update = max_pomo_idx < pomo_size
            incumbent_pomo_idx[inc_update] = max_pomo_idx[inc_update]

            # Loss: POMO RL
            ###############################################
            pomo_prob_list = prob_list[:, :pomo_size, :]
            # shape: (aug_batch, pomo, tour_len)
            pomo_reward = reward[:, :pomo_size]
            # shape: (aug_batch, pomo)

            advantage = pomo_reward - pomo_reward.float().mean(dim=1, keepdim=True)
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

            assert not torch.isnan(loss).any() #########################################
            loss.sum().backward()

            optimizer.step()

            for model in self.model.stage_models:
                assert not torch.isnan(model.decoder.eas_b2).any()


            # Score Curve
            ###############################################
            augbatch_score = incumbent_score.reshape(aug_factor, batch_size)
            # shape: (augmentation, batch)
            min_aug_score, _ = augbatch_score.min(dim=0)  # get best results from augmentation
            # shape: (batch,)
            sum_score = min_aug_score.sum()  # negative sign to make positive value
            sum_score_curve[iter_i] = sum_score


            ##################################################################################################################################
            self.logger.info('ep[{}]   EAS iteration: {}   score: {}'.format(episode, iter_i, sum_score.item()/batch_size))

        # Store Result Tensors in CPU_Memory
        ###############################################
        cpu_solution = incumbent_solution.reshape(aug_factor, batch_size, total_machine_cnt, job_cnt).to('cpu')
        self.all_incumbent_solution[:, episode:episode+batch_size] = cpu_solution
        cpu_score = incumbent_score.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_score[:, episode:episode+batch_size] = cpu_score
        cpu_pomo_idx = incumbent_pomo_idx.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_pomo_idx[:, episode:episode+batch_size] = cpu_pomo_idx

        for stage_idx in range(stage_cnt):
            model = self.model.stage_models[stage_idx]
            cpu_W1 = model.decoder.eas_W1.data.reshape(aug_factor, batch_size, emb_dim, emb_dim).to('cpu')
            self.all_eas_W1_list[stage_idx][:, episode:episode+batch_size] = cpu_W1
            cpu_b1 = model.decoder.eas_b1.data.reshape(aug_factor, batch_size, emb_dim).to('cpu')
            self.all_eas_b1_list[stage_idx][:, episode:episode+batch_size] = cpu_b1
            cpu_W2 = model.decoder.eas_W2.data.reshape(aug_factor, batch_size, emb_dim, emb_dim).to('cpu')
            self.all_eas_W2_list[stage_idx][:, episode:episode+batch_size] = cpu_W2
            cpu_b2 = model.decoder.eas_b2.data.reshape(aug_factor, batch_size, emb_dim).to('cpu')
            self.all_eas_b2_list[stage_idx][:, episode:episode+batch_size] = cpu_b2

        cpu_pomo_result = pomo_reward.reshape(aug_factor, batch_size, pomo_size).to('cpu')
        self.all_pomo_result[:, episode:episode+batch_size] = cpu_pomo_result

        return sum_score_curve

    ###########################################################################################################
    ###########################################################################################################
    # SGBS
    ###########################################################################################################
    ###########################################################################################################

    def _run_sgbs(self):

        # result_curve
        score_AM = AverageMeter()
        sgbs_step_max = self.run_params['sgbs_step_max']
        result_curve = torch.zeros(size=(sgbs_step_max,), dtype=torch.long)  # note: only the first element of the result curve is used at the moment

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

        # Done
        init_score = result_curve[0].float().item() / num_episode
        final_score = score_AM.avg
        return init_score, final_score

    def _sgbs_one_batch(self, episode, batch_size):

        aug_factor = self.run_params['aug_factor']
        job_cnt = self.env_params['job_cnt']
        machine_cnt_list = self.env_params['machine_cnt_list']
        total_machine_cnt = sum(machine_cnt_list)
        sgbs_step_max = self.run_params['sgbs_step_max']
        beam_width = self.run_params['beam_width']
        rollout_per_node = self.run_params['rollout_per_node']
        rollout_width = beam_width * rollout_per_node

        sum_score_curve = torch.zeros(size=(sgbs_step_max,), dtype=torch.long)

        # Ready
        ###############################################
        self._feed_loaded_data_in_batch(episode, batch_size, load_pomo_result=True)

        problems_INT_list = []
        for stage_idx in range(self.env.stage_cnt):
            problems_INT_list.append(self.ALL_problems_INT_list[stage_idx][episode:episode+batch_size])
            problems_INT_list[stage_idx] = problems_INT_list[stage_idx].repeat(aug_factor, 1, 1)
            # shape: (aug_batch_size, job_cnt, machine_cnt)

        self.env.load_problems_manual(problems_INT_list)

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.enable_EAS(True)
        self.model.pre_forward_w_saved_encodings(self.batch_encoded_row_list, self.batch_encoded_col_list)
        self.model.init_eas_layers_manual(self.batch_W1_list, self.batch_b1_list, self.batch_W2_list, self.batch_b2_list)
        self.model.requires_grad_(False)  # no grad on eas parameters

        # BS Step 1
        ###############################################
        bs_step = 1
        sorted_index = self.batch_pomo_result.argsort(dim=1, descending=True)
        beam_index = sorted_index[:, :beam_width]
        # shape: (aug*batch, beam_width)

        self.env.modify_pomo_size(beam_width)
        self.env.set_pomo_idx(beam_index)
        self.env.reset()

        state, _, _ = self.env.pre_step()
        selected, _ = self.model(state)
        # shape: (batch, beam_width)
        state, reward, done = self.env.step(selected)

        # Summed Scores
        aug_score = self.batch_incumbent_score.reshape(aug_factor, batch_size)
        # shape: (aug, batch)
        min_aug_score, _ = aug_score.min(dim=0)  # get best results from augmentation
        # shape: (batch,)
        sum_score_curve[bs_step-1] = min_aug_score.sum()  # first step

        # BS Step > 1
        ###############################################
        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.env)
        rollout_env.modify_pomo_size(rollout_width)

        # LOOP
        first_rollout_flag = True
        while not done:
            bs_step += 1

            # No Rollout-Based Beam Search after the limit
            if bs_step > sgbs_step_max:
                if bs_step == sgbs_step_max+1:
                    self.logger.info('ep[{}]  SGBS skipped from step {}'.format(episode, bs_step))

                # POMO Rollout
                ###############################################
                selected, _ = self.model(state)
                # shape: (batch, beam_width)
                state, reward, done = self.env.step(selected)
                continue

            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)
            # shape: (aug*batch, beam, job+1)
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

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.env, repeat=rollout_per_node)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_factor*batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                selected, _ = self.model(rollout_state)
                # shape: (aug*batch, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)
            # reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_factor*batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            max_makespan = job_cnt * 10
            rollout_reward[is_redundant] = -max_makespan

            # Merge Rollout-Env & BS-Env (Optional, for slightly better performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(self.env)
                rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                # reward.shape: (aug*batch, rollout_width + beam_width)
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

            # Score Curve
            ###############################################
            max_pomo_reward, _ = rollout_reward.max(dim=1)  # get best results from rollout
            # shape: (aug*batch,)
            min_pomo_score = -max_pomo_reward
            old_better = self.batch_incumbent_score < min_pomo_score
            min_pomo_score[old_better] = self.batch_incumbent_score[old_better]

            aug_score = min_pomo_score.reshape(aug_factor, batch_size)
            # shape: (aug, batch)
            min_aug_score, _ = aug_score.min(dim=0)  # get best results from augmentation
            # shape: (batch,)
            sum_score_curve[bs_step-1] = min_aug_score.sum()

            ##################################################################################################################################
            self.logger.info('ep[{}]   bs_step: {}   score: {}'.format(episode, bs_step, min_aug_score.sum().item()/batch_size))


        # Store Result Tensors in CPU_Memory
        ###############################################
        max_reward, max_idx = reward.max(dim=1)  # get best results from beam_width
        # shape: (aug_batch,)
        incumbent_score = -max_reward
        # shape: (aug_batch,)

        all_schedules = self.env.schedule[:, :, :, :job_cnt]
        # shape: (aug_batch, pomo, total_machine, job)
        gathering_index = max_idx[:, None, None, None].expand(-1, 1, total_machine_cnt, job_cnt)
        incumbent_solution = all_schedules.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (aug_batch, total_machine, job)

        incumbent_pomo_idx = self.env.pomo_idx[torch.arange(aug_factor*batch_size), max_idx]
        # shape: (aug_batch,)

        inc_update = self.batch_incumbent_score > incumbent_score
        self.batch_incumbent_score[inc_update] = incumbent_score[inc_update]
        self.batch_incumbent_solution[inc_update] = incumbent_solution[inc_update]
        self.batch_incumbent_pomo_idx[inc_update] = incumbent_pomo_idx[inc_update]

        cpu_score = self.batch_incumbent_score.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_score[:, episode:episode+batch_size] = cpu_score
        cpu_solution = self.batch_incumbent_solution.reshape(aug_factor, batch_size, total_machine_cnt, job_cnt).to('cpu')
        self.all_incumbent_solution[:, episode:episode+batch_size] = cpu_solution
        cpu_pomo_idx = self.batch_incumbent_pomo_idx.reshape(aug_factor, batch_size).to('cpu')
        self.all_incumbent_pomo_idx[:, episode:episode+batch_size] = cpu_pomo_idx

        # final_sum_score, sum_score_curve
        min_aug_score, _ = cpu_score.min(dim=0)
        return min_aug_score.sum().item(), sum_score_curve



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
import pickle

from TSPEnv import TSPEnv, get_random_problems, augment_xy_data_by_8_fold


class E_TSPEnv(TSPEnv):

    # def __init__(self, **model_params):
    #     super().__init__(**model_params)

    def load_problems_by_index(self, start_index, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.saved_index = start_index
            self.problems = self.saved_problems[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def step(self, selected):
        state, reward, done = super().step(selected)
        state.first_node = self.selected_node_list[:, :, 0]
        # shape: (batch, pomo)
        return state, reward, done

    def modify_pomo_size(self, new_pomo_size):
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        partial_data = list(data[i] for i in range(index_begin, index_begin+num_problems))

        self.saved_problems = torch.tensor(partial_data)
        self.saved_index = 0

    def copy_problems(self, old_env):
        self.batch_size = old_env.batch_size
        self.problems = old_env.problems

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask = bs_env.step_state.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, node)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size)
        self.step_state.ninf_mask = rollout_env.step_state.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem)

        if gathering_index.size(1) != self.pomo_size:
            self.modify_pomo_size(gathering_index.size(1))

    def merge(self, other_env):
        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo1 + pomo2)
        self.step_state.ninf_mask = torch.cat((self.step_state.ninf_mask, other_env.step_state.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem)

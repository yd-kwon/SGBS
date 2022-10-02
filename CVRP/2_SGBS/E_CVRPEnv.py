
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

from CVRPEnv import CVRPEnv, get_random_problems, augment_xy_data_by_8_fold

class E_CVRPEnv(CVRPEnv):

    # def __init__(self, **model_params):
    #     super().__init__(**model_params)

    def load_problems_by_index(self, start_index, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            self.saved_index = start_index
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

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

        depot_data = list(data[i][0] for i in range(index_begin, index_begin+num_problems))
        self.saved_depot_xy = torch.tensor(depot_data)[:, None, :]
        # shape: (batch, 1, 2)

        node_data = list(data[i][1] for i in range(index_begin, index_begin+num_problems))
        self.saved_node_xy = torch.tensor(node_data)
        # shape: (batch, problem, 2)

        demand_data = list(data[i][2] for i in range(index_begin, index_begin+num_problems))
        capacity_data = list(data[i][3] for i in range(index_begin, index_begin+num_problems))
        capacity_tensor = torch.tensor(capacity_data, dtype=torch.float)
        self.saved_node_demand = torch.tensor(demand_data, dtype=torch.float)/capacity_tensor[:, None]

        self.saved_index = 0

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = bs_env.at_the_depot.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.load = bs_env.load.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.visited_ninf_flag = bs_env.visited_ninf_flag.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = bs_env.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.finished = bs_env.finished.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = rollout_env.at_the_depot.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        self.load = rollout_env.load.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size+1)
        self.visited_ninf_flag = rollout_env.visited_ninf_flag.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = rollout_env.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.finished = rollout_env.finished.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)

        if gathering_index.size(1) != self.pomo_size:
            self.modify_pomo_size(gathering_index.size(1))

    def merge(self, other_env):

        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        self.at_the_depot = torch.cat((self.at_the_depot, other_env.at_the_depot), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.load = torch.cat((self.load, other_env.load), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.visited_ninf_flag = torch.cat((self.visited_ninf_flag, other_env.visited_ninf_flag), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.ninf_mask = torch.cat((self.ninf_mask, other_env.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.finished = torch.cat((self.finished, other_env.finished), dim=1)
        # shape: (batch, pomo1 + pomo2)

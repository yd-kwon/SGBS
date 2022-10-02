
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

from FFSPEnv import FFSPEnv, Step_State, Reset_State, _Stage_N_Machine_Index_Converter



class E_FFSPEnv(FFSPEnv):

    def __init__(self, **env_params):
        super().__init__(**env_params)
        self.pomo_idx = None
        # shape: (batch, pomo)
        # This is different from POMO_IDX. This keeps track of which type of pomo-env it is. (machines are assigned to jobs in different orders)
        # On the other hand, POMO_IDX is just for addressing in range

    def modify_pomo_size(self, new_pomo_size):
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        # STEP STATE
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def set_pomo_idx(self, pomo_idx):
        self.pomo_idx = pomo_idx

    def reset(self):  # Overwrite reset, to utilize self.pomo_idx
        # NOTE: Before using this reset function, one must set self.pomo_idx correctly!

        self.time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.sub_time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)

        self.machine_idx = self.sm_indexer.get_machine_index(self.pomo_idx, self.sub_time_idx)
        # shape: (batch, pomo)

        self.schedule = torch.full(size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt+1),
                                   dtype=torch.long, fill_value=-999999)
        # shape: (batch, pomo, machine, job+1)
        self.machine_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)
        self.job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.finished = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)
        # shape: (batch, pomo)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        reward = None
        done = None
        return Reset_State(self.problems_list), reward, done

    def _update_step_state(self):  # Overwrite, to utilize self.pomo_idx
        self.step_state.step_cnt += 1

        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        # shape: (batch, pomo)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.pomo_idx, self.sub_time_idx)
        # shape: (batch, pomo)

        job_loc = self.job_location[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)
        job_wait_t = self.job_wait_step[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)

        job_in_stage = job_loc == self.step_state.stage_idx[:, :, None]
        # shape: (batch, pomo, job)
        job_not_waiting = (job_wait_t == 0)
        # shape: (batch, pomo, job)
        job_available = job_in_stage & job_not_waiting
        # shape: (batch, pomo, job)

        job_in_previous_stages = (job_loc < self.step_state.stage_idx[:, :, None]).any(dim=2)
        # shape: (batch, pomo)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(dim=2)
        # shape: (batch, pomo)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished
        # shape: (batch, pomo)

        self.step_state.job_ninf_mask = torch.full(size=(self.batch_size, self.pomo_size, self.job_cnt+1),
                                                   fill_value=float('-inf'))
        # shape: (batch, pomo, job+1)
        job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
        # shape: (batch, pomo, job+1)
        self.step_state.job_ninf_mask[job_enable] = 0
        # shape: (batch, pomo, job+1)

        self.step_state.finished = self.finished
        # shape: (batch, pomo)

    def _move_to_next_machine(self):  # Overwrite, to utilize self.pomo_idx

        b_idx = torch.flatten(self.BATCH_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        p_idx = torch.flatten(self.POMO_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        ap_idx = torch.flatten(self.pomo_idx)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        ready = torch.flatten(self.finished)
        # shape: (batch*pomo,) == (not_ready_cnt,)

        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        p_idx = p_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        ap_idx = ap_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)

        while ~ready.all():
            new_sub_time_idx = self.sub_time_idx[b_idx, p_idx] + 1
            # shape: (not_ready_cnt,)
            step_time_required = new_sub_time_idx == self.total_machine_cnt
            # shape: (not_ready_cnt,)
            self.time_idx[b_idx, p_idx] += step_time_required.long()
            new_sub_time_idx[step_time_required] = 0
            self.sub_time_idx[b_idx, p_idx] = new_sub_time_idx
            new_machine_idx = self.sm_indexer.get_machine_index(ap_idx, new_sub_time_idx)
            self.machine_idx[b_idx, p_idx] = new_machine_idx

            machine_wait_steps = self.machine_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.machine_wait_step[b_idx, p_idx, :] = machine_wait_steps

            job_wait_steps = self.job_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.job_wait_step[b_idx, p_idx, :] = job_wait_steps

            machine_ready = self.machine_wait_step[b_idx, p_idx, new_machine_idx] == 0
            # shape: (not_ready_cnt,)

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)
            # shape: (not_ready_cnt,)
            job_ready_1 = (self.job_location[b_idx, p_idx, :self.job_cnt] == new_stage_idx[:, None])
            # shape: (not_ready_cnt, job)
            job_ready_2 = (self.job_wait_step[b_idx, p_idx, :self.job_cnt] == 0)
            # shape: (not_ready_cnt, job)
            job_ready = (job_ready_1 & job_ready_2).any(dim=1)
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            p_idx = p_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            ap_idx = ap_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)

    ###########################################################################################################
    ###########################################################################################################
    # For EAS
    ###########################################################################################################
    ###########################################################################################################

    def extract_action_from_schedule(self, schedule):
        # schedule.shape: (batch, total_machine_cnt, job_cnt)
        # Note: we use last element of the pomo dimension for env

        time_idx = self.time_idx[:, -1]
        # shape: (batch,)
        machine_idx = self.machine_idx[:, -1]
        # shape: (batch,)

        gathering_index = machine_idx[:, None, None].expand(-1, 1, self.job_cnt)
        current_machine_schedule = schedule.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (batch, job_cnt)
        (batch_id, job_id) = torch.nonzero(current_machine_schedule == time_idx[:, None], as_tuple=True)

        no_job_idx = self.env_params['job_cnt']
        action = torch.full_like(time_idx, fill_value=no_job_idx)
        action[batch_id] = job_id

        # if finished => no job
        action[self.finished[:, -1]] = no_job_idx

        return action

    ###########################################################################################################
    ###########################################################################################################
    # For SGBS
    ###########################################################################################################
    ###########################################################################################################

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.pomo_idx = bs_env.pomo_idx.repeat_interleave(repeat, dim=1)

        self.time_idx = bs_env.time_idx.repeat_interleave(repeat, dim=1)
        self.sub_time_idx = bs_env.sub_time_idx.repeat_interleave(repeat, dim=1)
        self.machine_idx = bs_env.machine_idx.repeat_interleave(repeat, dim=1)

        self.schedule = bs_env.schedule.repeat_interleave(repeat, dim=1)
        self.machine_wait_step = bs_env.machine_wait_step.repeat_interleave(repeat, dim=1)
        self.job_location = bs_env.job_location.repeat_interleave(repeat, dim=1)
        self.job_wait_step = bs_env.job_wait_step.repeat_interleave(repeat, dim=1)
        self.finished = bs_env.finished.repeat_interleave(repeat, dim=1)

        # STEP STATE
        self.step_state.step_cnt = bs_env.step_state.step_cnt  # step_cnt seems not needed at all
        self.step_state.stage_idx = bs_env.step_state.stage_idx.repeat_interleave(repeat, dim=1)
        self.step_state.stage_machine_idx = bs_env.step_state.stage_machine_idx.repeat_interleave(repeat, dim=1)
        self.step_state.job_ninf_mask = bs_env.step_state.job_ninf_mask.repeat_interleave(repeat, dim=1)
        self.step_state.finished = self.finished

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        # gathering_index.shape: (aug_batch, beam_width)

        self.pomo_idx = rollout_env.pomo_idx.gather(dim=1, index=gathering_index)

        self.time_idx = rollout_env.time_idx.gather(dim=1, index=gathering_index)
        self.sub_time_idx = rollout_env.sub_time_idx.gather(dim=1, index=gathering_index)
        self.machine_idx = rollout_env.machine_idx.gather(dim=1, index=gathering_index)

        exp_gathering_index = gathering_index[:, :, None, None].expand(-1, -1, self.total_machine_cnt, self.job_cnt+1)
        self.schedule = rollout_env.schedule.gather(dim=1, index=exp_gathering_index)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.total_machine_cnt)
        self.machine_wait_step = rollout_env.machine_wait_step.gather(dim=1, index=exp_gathering_index)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.job_cnt+1)
        self.job_location = rollout_env.job_location.gather(dim=1, index=exp_gathering_index)
        self.job_wait_step = rollout_env.job_wait_step.gather(dim=1, index=exp_gathering_index)
        self.finished = rollout_env.finished.gather(dim=1, index=gathering_index)

        # STEP STATE
        self.step_state.step_cnt = rollout_env.step_state.step_cnt
        self.step_state.stage_idx = rollout_env.step_state.stage_idx.gather(dim=1, index=gathering_index)
        self.step_state.stage_machine_idx = rollout_env.step_state.stage_machine_idx.gather(dim=1, index=gathering_index)
        self.step_state.job_ninf_mask = rollout_env.step_state.job_ninf_mask.gather(dim=1, index=exp_gathering_index)
        self.step_state.finished = self.finished

    def merge(self, other_env):
        self.pomo_idx = torch.cat((self.pomo_idx, other_env.pomo_idx), dim=1)

        self.time_idx = torch.cat((self.time_idx, other_env.time_idx), dim=1)
        self.sub_time_idx = torch.cat((self.sub_time_idx, other_env.sub_time_idx), dim=1)
        self.machine_idx = torch.cat((self.machine_idx, other_env.machine_idx), dim=1)

        self.schedule = torch.cat((self.schedule, other_env.schedule), dim=1)
        self.machine_wait_step = torch.cat((self.machine_wait_step, other_env.machine_wait_step), dim=1)
        self.job_location = torch.cat((self.job_location, other_env.job_location), dim=1)
        self.job_wait_step = torch.cat((self.job_wait_step, other_env.job_wait_step), dim=1)
        self.finished = torch.cat((self.finished, other_env.finished), dim=1)

        # STEP STATE
        self.step_state.stage_idx = torch.cat((self.step_state.stage_idx, other_env.step_state.stage_idx), dim=1)
        self.step_state.stage_machine_idx = torch.cat((self.step_state.stage_machine_idx, other_env.step_state.stage_machine_idx), dim=1)
        self.step_state.job_ninf_mask = torch.cat((self.step_state.job_ninf_mask, other_env.step_state.job_ninf_mask), dim=1)
        self.step_state.finished = self.finished


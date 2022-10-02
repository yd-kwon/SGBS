
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
import torch.nn as nn
import torch.nn.functional as F

from FFSPModel import FFSPModel, FFSP_Encoder, FFSP_Decoder, OneStageModel, reshape_by_heads


class E_FFSPModel(FFSPModel):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        stage_cnt = self.model_params['stage_cnt']
        self.stage_models = nn.ModuleList([E_OneStageModel(stage_idx, **model_params) for stage_idx in range(stage_cnt)])

    def pre_forward_w_saved_encodings(self, encoded_row_list, encoded_col_list):
        stage_cnt = self.model_params['stage_cnt']
        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            encoded_row = encoded_row_list[stage_idx]
            encoded_col = encoded_col_list[stage_idx]
            model.pre_forward_w_saved_encodings(encoded_row, encoded_col)

    def enable_EAS(self, bool):
        stage_cnt = self.model_params['stage_cnt']
        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            model.decoder.enable_EAS = bool

    def init_eas_layers_manual(self, W1_list, b1_list, W2_list, b2_list):
        stage_cnt = self.model_params['stage_cnt']
        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            W1 = W1_list[stage_idx]
            b1 = b1_list[stage_idx]
            W2 = W2_list[stage_idx]
            b2 = b2_list[stage_idx]
            model.decoder.init_eas_layers_manual(W1, b1, W2, b2)

    def eas_parameters(self):
        stage_cnt = self.model_params['stage_cnt']
        params = []
        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            params += model.decoder.eas_parameters()
        return params

    def forward_w_incumbent(self, state, best_action):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        stage_cnt = self.model_params['stage_cnt']
        action_stack = torch.empty(size=(batch_size, pomo_size, stage_cnt), dtype=torch.long)
        prob_stack = torch.empty(size=(batch_size, pomo_size, stage_cnt))

        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            action, prob = model.forward_w_incumbent(state, best_action)

            action_stack[:, :, stage_idx] = action
            prob_stack[:, :, stage_idx] = prob

        gathering_index = state.stage_idx[:, :, None]
        # shape: (batch, pomo, 1)
        action = action_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        prob = prob_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        return action, prob

    def get_expand_prob(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        job_cnt_p1 = state.job_ninf_mask.size(2)

        stage_cnt = self.model_params['stage_cnt']
        probs_stack = torch.empty(size=(batch_size, pomo_size, job_cnt_p1, stage_cnt))

        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            probs = model.get_expand_prob(state)

            probs_stack[:, :, :, stage_idx] = probs

        gathering_index = state.stage_idx[:, :, None, None].expand(batch_size, pomo_size, job_cnt_p1, 1)
        # shape: (batch, pomo, job_cnt+1, 1)

        probs = probs_stack.gather(dim=3, index=gathering_index).squeeze(dim=3)
        # shape: (batch, pomo, job_cnt+1)

        return probs


class E_OneStageModel(OneStageModel):

    def __init__(self, stage_idx, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = FFSP_Encoder(**model_params)
        self.decoder = E_FFSP_Decoder(**model_params)

        self.encoded_col = None
        # shape: (batch, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch, job_cnt, embedding)

    def pre_forward_w_saved_encodings(self, encoded_row, encoded_col):

        self.encoded_row = encoded_row
        self.encoded_col = encoded_col
        # encoded_row.shape: (batch, job_cnt, embedding)
        # encoded_col.shape: (batch, machine_cnt, embedding)

        self.decoder.set_kv(self.encoded_row)

    def forward_w_incumbent(self, state, best_action):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)

        encoded_current_machine = self._get_encoding(self.encoded_col, state.stage_machine_idx)
        # shape: (batch, pomo+1, embedding)
        all_job_probs = self.decoder(encoded_current_machine,
                                     ninf_mask=state.job_ninf_mask)
        # shape: (batch, pomo+1, job+1)

        if self.training or self.model_params['eval_type'] == 'softmax':
            while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                job_selected = all_job_probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                # shape: (batch, pomo+1)
                job_selected[:, -1] = best_action
                job_prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, job_selected] \
                    .reshape(batch_size, pomo_size_p1)
                # shape: (batch, pomo+1)
                job_prob[state.finished] = 1  # do not backprob finished episodes

                if (job_prob[:, :-1] != 0).all():
                    break
        else:
            job_selected = all_job_probs.argmax(dim=2)
            # shape: (batch, pomo+1)
            job_selected[:, -1] = best_action
            job_prob = None  # any number is okay

        return job_selected, job_prob

    def get_expand_prob(self, state):
        encoded_current_machine = self._get_encoding(self.encoded_col, state.stage_machine_idx)
        # shape: (batch, pomo+1, embedding)
        all_job_probs = self.decoder(encoded_current_machine,
                                     ninf_mask=state.job_ninf_mask)
        # shape: (batch, pomo, job+1)

        return all_job_probs

class E_FFSP_Decoder(FFSP_Decoder):

    def __init__(self, **model_params):
        super().__init__(**model_params)

        self.enable_EAS = None  # bool

        self.eas_W1 = None
        # shape: (batch, embedding, embedding)
        self.eas_b1 = None
        # shape: (batch, embedding)
        self.eas_W2 = None
        # shape: (batch, embedding, embedding)
        self.eas_b2 = None
        # shape: (batch, embedding)

    def init_eas_layers_manual(self, W1, b1, W2, b2):
        self.eas_W1 = torch.nn.Parameter(W1)
        self.eas_b1 = torch.nn.Parameter(b1)
        self.eas_W2 = torch.nn.Parameter(W2)
        self.eas_b2 = torch.nn.Parameter(b2)

    def eas_parameters(self):
        return [self.eas_W1, self.eas_b1, self.eas_W2, self.eas_b2]

    def forward(self, encoded_machine, ninf_mask):
        # encoded_machine.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job_cnt+1)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v,
                                                            rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # EAS Layer Insert
        #######################################################
        if self.enable_EAS:
            ms1 = torch.matmul(mh_atten_out, self.eas_W1)
            # shape: (batch, pomo, embedding)

            ms1 = ms1 + self.eas_b1[:, None, :]
            # shape: (batch, pomo, embedding)

            ms1_activated = F.relu(ms1)
            # shape: (batch, pomo, embedding)

            ms2 = torch.matmul(ms1_activated, self.eas_W2)
            # shape: (batch, pomo, embedding)

            ms2 = ms2 + self.eas_b2[:, None, :]
            # shape: (batch, pomo, embedding)

            mh_atten_out = mh_atten_out + ms2
            # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job_cnt+1)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job_cnt+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job_cnt+1)

        return probs


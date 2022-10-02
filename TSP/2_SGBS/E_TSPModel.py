
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

from TSPModel import TSPModel, _get_encoding, TSP_Encoder, TSP_Decoder, reshape_by_heads, multi_head_attention

class E_TSPModel(TSPModel):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = E_TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def pre_forward(self, reset_state):  # now includes decoder.set_q1
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

        batch_size = reset_state.problems.size(0)
        problem_size = reset_state.problems.size(1)
        all_nodes = torch.arange(problem_size)[None, :].expand(batch_size, problem_size)
        encoded_first_node = _get_encoding(self.encoded_nodes, all_nodes)
        # shape: (batch, pomo, embedding)
        self.decoder.set_q1(encoded_first_node)

    def get_expand_prob(self, state):

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        # shape: (batch, beam_width, embedding)
        probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node)
        # shape: (batch, beam_width, problem)

        return probs

    def forward(self, state, eval_type='greedy'):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node)
            # shape: (batch, pomo, problem)

            if self.training or eval_type == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

    def forward_w_incumbent(self, state, best_action):
        # best_action.shape = (batch,)
        # first_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1-1

        if state.current_node is None:  # First Move, POMO
            pomo_select = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            # shape: (batch, pomo)
            selected = torch.cat((pomo_select, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node)
            # shape: (batch, pomo+1, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                    # shape: (batch, pomo+1)
                    selected[:, -1] = best_action
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size_p1)
                    # shape: (batch, pomo+1)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                selected[:, -1] = best_action
                prob = None  # value not needed. Can be anything.

        return selected, prob


class E_TSP_Decoder(TSP_Decoder):

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
        
    def init_eas_layers_random(self, batch_size):
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1/emb_dim)**(1/2)
        
        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim))
        self.eas_W1 = torch.nn.Parameter(weight1)
        self.eas_b1 = torch.nn.Parameter(bias1)
        self.eas_W2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim, emb_dim)))
        self.eas_b2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim)))

    def init_eas_layers_manual(self, W1, b1, W2, b2):
        self.eas_W1 = torch.nn.Parameter(W1)
        self.eas_b1 = torch.nn.Parameter(b1)
        self.eas_W2 = torch.nn.Parameter(W2)
        self.eas_b2 = torch.nn.Parameter(b2)

    def eas_parameters(self):
        return [self.eas_W1, self.eas_b1, self.eas_W2, self.eas_b2]
        
    def forward(self, encoded_last_node, ninf_mask, first_node=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        # first_node.shape: (batch, modified_pomo)  # use first_node=None when pomo = {1, 2, ..., problem}

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        if first_node is None:
            q_first = self.q_first
            # shape: (batch, head_num, pomo, qkv_dim)
        else:
            qkv_dim = self.model_params['qkv_dim']
            gathering_index = first_node[:, None, :, None].expand(-1, head_num, -1, qkv_dim)
            q_first = self.q_first.gather(dim=2, index=gathering_index)
            # shape: (batch, head_num, mod_pomo, qkv_dim)

        q = q_first + q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
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
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs






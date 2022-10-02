
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

from CVRPModel import CVRPModel, _get_encoding, CVRP_Encoder, CVRP_Decoder, reshape_by_heads, multi_head_attention

class E_CVRPModel(CVRPModel):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = EAS_CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def get_expand_prob(self, state):

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        # shape: (batch, beam_width, embedding)
        probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
        # shape: (batch, beam_width, problem+1)

        return probs

    def forward_w_incumbent(self, state, best_action):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1-1

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size_p1), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            selected = torch.cat((selected, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
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


    def forward_w_incumbent_probs(self, state, best_action):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1-1

        probs = None
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size_p1), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            selected = torch.cat((selected, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
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

        return selected, prob, probs


class EAS_CVRP_Decoder(CVRP_Decoder):

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
        
    def forward(self, encoded_last_node, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
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






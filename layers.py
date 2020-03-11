import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttention2(nn.Module):

  def __init__(self, query_dim):
    super(SelfAttention2, self).__init__()
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
    keys = keys.transpose(1,2) # [TxBxK] -> [BxKxT]
    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        # https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter#answer-51027227
        self.v    = nn.Parameter(torch.rand(self.hidden_size))
        stdv      = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        h = hidden.repeat(max_len, 1, 1).transpose(0, 1) # repeats the hidden along the (max_len, 1, 1) times.
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        softmax = F.softmax(attn_energies, dim=1).unsqueeze(1)
        return softmax

    def score(self, hidden, encoder_outputs):
        # batch_size X seq_len X 2 * hidden_size ==> batch_size X seq_len X hidden_size
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = energy.transpose(1, 2) # batch_size X hidden_size X seq_len
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1) # batch_size X 1 X hidden_size
        energy = torch.bmm(v, energy) # batch_size X 1 X seq_len
        return energy.squeeze(1) # batch_size X seq_len

class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size, hidden_size, embedding_dim, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size         = hidden_size
        self.output_size         = output_size
        self.n_layers            = n_layers
        self.embedding_dim       = embedding_dim

        self.attention = Attention(encoder_hidden_size + hidden_size)
        self.gru       = nn.GRU(encoder_hidden_size + embedding_dim, hidden_size, n_layers, dropout=dropout)
        self.out       = nn.Linear(encoder_hidden_size + hidden_size, output_size)

    def forward(self, embedded, last_hidden, encoder_outputs):
        # calculate the attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context      = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # batch_size X 1 X N
        context      = context.transpose(0, 1) # 1 X batch_size X N

        # combine embedded input word and attended context
        rnn_input    = torch.cat([embedded, context], 2)

        # run through the GRU layer
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0) # 1 X batch_size X N ==> batch_size X N
        context = context.squeeze(0)

        # concat output and context
        cat_output_context = torch.cat([output, context], 1)

        # run through the linear layer
        output = self.out(cat_output_context)

        # find the log softmax
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class SelfAttention(nn.Module):
    def __init__(self, opt, attention_size,
                 layers=2,
                 dropout=0.1,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.opt = opt

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if self.opt['cuda']:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        # return representations, scores

        return weighted

class SingleEncoder(nn.Module):

    def __init__(self, opt, input_size):
        super(SingleEncoder, self).__init__()
        self.opt = opt
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=input_size,
                           num_layers=opt['encoder_layer'],
                           bidirectional=True)

    def forward(self, inputs):
        # inputs = batch_size X seq_len X hidden_size
        # out = batch_size X (seq_len * hidden_size)

        inputs = inputs.transpose(1, 0)

        batch_size = inputs.size()[1]
        hidden_size = inputs.size()[2]
        state_shape = self.opt['encoder_layer']*2, batch_size, hidden_size
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return outputs.transpose(1, 0), ht

class StackedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False):
        super(StackedEncoder, self).__init__()
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        # Transpose batch and sequence dims
        # x: batch_size X seq_len X hidden_size
        # output: batch_size X seq_len X hidden_size

        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)

        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.contiguous().view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.contiguous().view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq
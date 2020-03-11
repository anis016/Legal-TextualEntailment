import torch
import torch.nn as nn
import layers
import numpy as np
import torch.nn.functional as F


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.contiguous().view(size[0]*size[1], -1))

        return out.view(size[0], size[1], -1)

class Linear(Bottle, nn.Linear):
    pass

class ArticleReader(nn.Module):
    """Network for the Article and Query Reader module of LegalQAClassifier."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(ArticleReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False

        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        # Projection for attention weighted query
        if opt['use_t2_emb']:
            self.t2emb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + query emb + manual features
        article_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_t2_emb']:
            article_input_size += opt['embedding_dim']
        if opt['pos']:
            article_input_size += opt['pos_size']
        if opt['ner']:
            article_input_size += opt['ner_size']

        # self.self_attn = layers.SelfAttention(opt, opt['embedding_dim'])
        # doc_input_size += opt['embedding_dim']

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=opt['dropout_linear'])

        # Stacked LSTM article encoder
        self.article = layers.StackedEncoder(
            input_size=article_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['t1_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']]
        )

        # Stacked LSTM query encoder
        self.query = layers.StackedEncoder(
            input_size=opt['embedding_dim'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['t2_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']]
        )

        # Output sizes of Stacked LSTM Encoders
        article_hidden_size = 2 * opt['hidden_size'] # article_hidden_size = 256
        query_hidden_size = 2 * opt['hidden_size']   # query_hidden_size = 256
        if opt['concat_rnn_layers']:
            article_hidden_size *= opt['t1_layers'] # article_hidden_size = 768
            query_hidden_size *= opt['t2_layers']   # query_hidden_size = 768

        self.single_encoder = layers.SingleEncoder(opt, article_hidden_size)

        # self.decoder = layers.Decoder(2 * article_hidden_size, article_hidden_size, opt['embedding_dim'],
        #                               self.opt['vocab_size'], n_layers=2)

        # encoders_dim = 2 * article_hidden_size  # seq_in_size = 1536
        # self.attention = layers.Attention2(encoders_dim)

        seq_in_size = 2 * article_hidden_size # seq_in_size = 1536, as encoder is bi-directional
        merge_size  = 2 * seq_in_size #  merging both encoder, seq_in_size = 3072
        lin_config = [merge_size] * 2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(merge_size, opt['d_out'])
        )

    def forward(self, t1, t1_f, t1_pos, t1_ner, t1_mask, t2, t2_mask):
        """Inputs:
        t1      = article word indices          [batch * len_d]
        t1_f    = article word features indices [batch * len_d * nfeat]
        t1_pos  = article POS tags              [batch * len_d]
        t1_ner  = article entity tags           [batch * len_d]
        t1_mask = article padding mask          [batch * len_d]
        t2      = query word indices            [batch * len_q]
        t2_mask = query padding mask            [batch * len_q]
        """
        # Embed both article and query
        t1_emb = self.embedding(t1) # batch_size X seq_len X emb_dim
        t2_emb = self.embedding(t2) # batch_size X seq_len X emb_dim

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            t1_emb = nn.functional.dropout(t1_emb,
                                           p=self.opt['dropout_emb'],
                                           training=self.training)

            t2_emb = nn.functional.dropout(t2_emb,
                                           p=self.opt['dropout_emb'],
                                           training=self.training)

        article_input_list = [t1_emb, t1_f]

        # Add attention-weighted query representation
        if self.opt['use_t2_emb']:
            t2_weighted_emb = self.t2emb_match(t1_emb, t2_emb, t2_mask)
            article_input_list.append(t2_weighted_emb) # batch_size X t1_seq_len X emb_dim
        if self.opt['pos']:
            article_input_list.append(t1_pos)
        if self.opt['ner']:
            article_input_list.append(t1_ner)

        # lengths = t1_mask.data.eq(0).long().sum(1).squeeze()
        # t1_self = self.self_attn(t1_emb, lengths)
        # article_input_list.append(t1_self)

        article_input = torch.cat(article_input_list, 2) # batch_size X t1_seq_len X (emb_dim * 2 + 4)

        # Encode article with Stacked LSTM
        article_hiddens = self.article(article_input, t1_mask) # batch_size X seq_len X hidden_size

        # Encode query with Stacked LSTM
        query_hiddens = self.query(t2_emb, t2_mask) # batch_size X seq_len X hidden_size

        ###
        # length_article = t1_mask.data.eq(0).long().sum(1).squeeze()
        # length_query = t2_mask.data.eq(0).long().sum(1).squeeze()
        # article_encoder, _ = self.self_attn(article_hiddens, length_article)
        # query_encoder, _ = self.self_attn(query_hiddens, length_query)
        ###

        art_out, art_ht = self.single_encoder(article_hiddens)
        que_out, que_ht = self.single_encoder(query_hiddens)

        # art_out = batch_size X t1_seq_len X 2 * hidden_size
        # que_out = batch_size X t2_seq_len X 2 * hidden_size
        # art_ht = que_ht = num_direction X batch_size X hidden_size

        # merge hiddens for the last encoder
        # article_encoder = batch_size X (2 * article_hidden_size)
        # query_encoder   = batch_size X (2 * query_hidden_size)
        batch_size = art_ht.shape[1]
        article_encoder = art_ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        query_encoder   = que_ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        concat_article_query = torch.cat([article_encoder, query_encoder], 1)
        scores = self.out(concat_article_query)

        if self.opt['interact']:

            dot_ = np.dot(art_out[0], que_out[0].transpose(1, 0))
            max_ = np.max(dot_)
            normalize_ = dot_ / max_

            return scores, normalize_

        return scores
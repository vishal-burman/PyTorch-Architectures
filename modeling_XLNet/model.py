import torch
import torch.nn as nn
import torch.nn.functional as F

class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)
        
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self):
        pass

class XLNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.seq_len_dim = 1

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r)

class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = clamp_len
        self.n_layer = config.n_layer
        
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in config.n_layers])
        self.dropout = nn.Dropout(config.dropout)

    def positional_embedding(self, pos_seq, inv_freq, bs=None): # pos_seq, inv_freq ~ [2 * max_len]
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq) # sinusoid_inp ~ [2 * max_len, d_model / 2]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1) # pos_emb ~ [2 * max_len, d_model]
        pos_emb = pos_emb[:, None, :] # pos_emb ~ [2 * max_len, 1, d_model]
        pos_emb = pos_emb.expand(-1, bs, -1) # pos_emb ~ [2 * max_len, bs // 2, d_model]
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bs=None): # qlen = klen
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float) # freq_seq ~ [d_model / 2]
        inv_seq = 1 / torch.pow(10000, (freq_seq / self.d_model)) # inv_seq ~ [d_model / 2]
        beg, end = klen, -qlen
        fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float) # fwd_pos_emb ~ [2 * max_len]
        bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float) # bwd_pos_emb ~ [2 * max_len] 
        fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bs // 2) # fwd_pos_emb ~ [2 * max_len, bs//2, d_model]
        bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bs // 2) # bwd_pos_emb ~ [2 * max_len, bs//2, d_model]
        pos_emb = torch.cat([fwd_pos_emb bwd_pos_emb], dim=1) # pos_emb ~ [2 * max_len, bs, d_model]
        return pos_emb


    def forward(self, input_ids=None, attention_mask=None): # input_ids, attention_mask ~ [batch_size, max_len]
        input_ids = input_ids.transpose(0, 1).contiguous() # input_ids ~ [max_len, batch_size]
        qlen, bs = input_ids.size(0), input_ids.size(1)
        attention_mask = attention_mask.transpose(0, 1).contiguous() # attention_mask ~ [max_len, batch_size]
        mlen = 0
        klen = mlen + qlen # klen = max_len
        attn_mask = None
        input_mask = 1.0 - attention_mask # input_mask ~ [max_len, batch_size]
        data_mask = input_mask[None] # data_mask ~ [1, max_le, batch_size]
        if attn_mask is None:
            attn_mask = data_mask[:, :, :, None]  # attn_mask ~ [1, max_len, batch_size, 1]
        attn_mask = (attn_mask > 0).to(torch.float) # attn_mask ~ [1, max_len, batch_size, 1]
        non_tgt_mask = -torch.eye(qlen).to(attn_mask) # non_tgt_mask ~ [max_len, max_len]
        non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask) # non_tgt_mask ~ [max_len, max_len, batch_size, 1]
        word_emb_k = self.word_embedding(input_ids) # word_emb_k ~ [max_len, batch_size, emb_size]
        output_h = self.dropout(word_emb_k) # output_h ~ [max_len, batch_size, emb_size]
        output_g = None
        seg_mat = None
        pos_emb = self.relative_positional_encoding(qlen, klen, bs=bs) # pos_emb ~ [2 * max_len, bs, d_model]
        pos_emb = self.dropout(pos_emb) # pos_emb ~ [2 * max_len, bs, d_model]
        for i, layer_module in enumerate(self.layer):
            outputs = layer_module(output_h,
                    output_g,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    r=pos_emb,
                    )


class XLNetClassify(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_type = config.attn_type
        self.same_length =config.same_length
        
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        input_ids ~ [batch_size, max_seq_len]
        attention_mask ~ [batch_size, max_seq_len]
        """
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        return transformer_outputs

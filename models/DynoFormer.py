import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.Embed import DataEmbedding_inverted
from layers.Embedding import positional_encoding
from layers.RevIN import RevIN
from functools import reduce
from operator import mul
import torch.nn.functional as F

from layers.StandardNorm import Normalize


def FFT_for_Period(x, k=2):
    r"""
    FFT for periodic -- TimesNet
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights

class Period_Conv_Attention(nn.Module):
    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()



        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v


        self.W_Q = nn.Conv1d(d_model,d_model, kernel_size=1,stride=1)
        self.W_K = nn.Conv1d(d_model,d_model, kernel_size=1,stride=1)
        self.W_V = nn.Conv1d(d_model,d_model, kernel_size=1,stride=1)
        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention

        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads, n_heads), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        q_b, q_nvar, q_dim, q_period_num, q_paroid_len = Q.shape
        Q = torch.reshape(Q, (q_b * q_nvar, q_dim, q_period_num, q_paroid_len))
        Q = torch.reshape(Q, (q_b * q_nvar, q_dim, q_period_num * q_paroid_len))

        k_b, k_nvar, k_dim, k_period_num, k_paroid_len = K.shape
        K = torch.reshape(K, (k_b * k_nvar, k_dim, k_period_num, k_paroid_len))
        K = torch.reshape(K, (k_b * k_nvar, k_dim, k_period_num * k_paroid_len))

        v_b, v_nvar, v_dim, v_period_num, v_paroid_len = V.shape
        V = torch.reshape(V, (v_b * v_nvar, v_dim, v_period_num, v_paroid_len))
        V = torch.reshape(V, (v_b * v_nvar, v_dim, v_period_num * v_paroid_len))

        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        w_q_s = self.W_Q(Q)
        w_k_s = self.W_K(K)
        w_v_s = self.W_V(V)
        q_s = w_q_s.view(q_b * q_nvar, q_dim, q_period_num, q_paroid_len).permute(0, 1, 3,
                                                                                  2)  # q_s    : [bs x n_heads x q_len x d_k]  此处的q_len为patch_num
        k_s = w_k_s.view(k_b * k_nvar, k_dim, k_period_num,
                         k_paroid_len)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = w_v_s.view(v_b * v_nvar, v_dim, v_period_num, v_paroid_len).permute(0, 1, 3,
                                                                                  2)  # q_s    : [bs x n_heads x q_len x d_k]  此处的q_len为patch_num

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        bso, dim, period_num, period_len = output.shape
        output = output.reshape(bso, dim, period_num * period_len)[:, :, :self.n_heads].contiguous()
        output = self.to_out(output)

        return output, attn_weights
class LocalGlobalEncoder(nn.Module):
    def __init__(self,configs):
        super(LocalGlobalEncoder, self).__init__()
        self.configs = configs

        self.activation = nn.ReLU()
        self.up_sampling_encoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.d_model,
                    configs.d_model,
                )
                for i in range(configs.down_sampling_layers +1 )
            ]
        )


    def forward(self,B, enc_out_list, x_list):
        dec_out_list = []
        x_list = x_list[0]
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            # dec_out = F.interpolate(enc_out.permute(0, 2, 1), size=self.configs.d_model, mode='linear', align_corners=False).permute(
            #     0, 2, 1)  # align temporal dimension
            # dec_out = self.up_sampling_encoder(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            if self.configs.use_linear == 1:
                dec_out = self.up_sampling_encoder[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            else:
                dec_out = enc_out
            dec_out = dec_out.reshape(B, self.configs.enc_in, self.configs.d_model).permute(0, 2,1).contiguous()
            # dec_out = dec_out.reshape(B, self.configs.enc_in, self.configs.d_model).permute(0, 2,1).contiguous()
            if self.configs.use_relu == 1:
                dec_out = self.activation(dec_out)
            dec_out_list.append(dec_out)
            # dec_out_list.append(self.activation(dec_out))
        return dec_out_list

class DynoFormerLayer(nn.Module):
    def __init__(self,configs):
        super(DynoFormerLayer, self).__init__()
        self.configs = configs
        self.k = configs.k
        self.d_ff = configs.d_ff
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.batch_size = configs.batch_size
        self.c_out = configs.c_out

        # embedding
        self.inverted_embedding = DataEmbedding_inverted(self.configs.seq_len, self.configs.d_model, self.configs.embed, self.configs.freq,
                                                    configs.dropout)

        self.start_embed = nn.Linear(1,self.d_model)

        self.de_embed = nn.Linear(self.d_model,self.seq_len)
        self.de_embed_fourier = nn.Linear(self.d_model,self.seq_len)
        self.un_embed_dim = nn.Linear(self.d_model,1)

        # period attention
        self.inter_d_model = self.d_model
        n_heads = self.d_model

        d_k = self.d_model * self.d_model
        d_v = self.d_model * self.d_model

        ##inter_embedding
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)


        self.period_attention =Period_Conv_Attention(self.inter_d_model,self.inter_d_model,n_heads,d_k, d_v, attn_dropout=0,
                                          proj_dropout=0.1, res_attention=False)


        #  fourier
        self.local_global_enc = LocalGlobalEncoder(configs)
        self.channel_independence = self.configs.channel_independence
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True,
                          non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(self.seq_len, self.d_model, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_model, self.seq_len, bias=True))


    def divide_into_segments(self,distance, divide_len):
        segment_size = distance / divide_len
        segments = []
        current_value = 0
        for _ in range(divide_len):
            segment = current_value
            segments.append(segment)
            current_value += segment_size
        return segments

    def fourier_zero(self, x_enc,time_step,point_frequence):

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_sampling_list = []
        if self.configs.use_origin_seq == 1:
            x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        sample_rate = 4096*1

        freq, _ = torch.sort(abs(torch.fft.fftfreq(self.configs.d_model, 1 / sample_rate)))
        min_freq = freq[0]
        max_freq = freq[-1]
        distance = abs(max_freq - min_freq).item()
        divide_len_list = self.divide_into_segments(distance,divide_len = self.configs.down_sampling_layers)

        fft_signal = torch.fft.fft(x_enc, dim=2, norm='ortho')  # signal_data为长度8192的list

        for index in range(len(divide_len_list)):
            cut_fft = fft_signal.clone()
            cut_fft[:, :,(freq < divide_len_list[index])] = 0
            if index + 1 < len(divide_len_list):
                cut_fft[:, :,(freq > divide_len_list[index + 1])] = 0

            cut_fft_signal = torch.fft.ifft(cut_fft,dim=2, norm='ortho')
            x_enc_sampling_list.append(cut_fft_signal.real.permute(0, 2, 1).contiguous())

        return x_enc_sampling_list
    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)
    def forward(self,x):
        new_x = x # b* seq_len * nvar
        x_embed = self.inverted_embedding(x,None).permute(0,2,1).contiguous()
        x_fourier_embed = x_embed

        B, T, N = x_embed.size()  # T 128 N 7  B 32
        period_list, period_weight = FFT_for_Period(x_embed, self.k)  # 傅里叶快速变换
        x_embed = self.start_embed(x_embed.unsqueeze(-1))
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.d_model % period != 0:
                length = period - (self.d_model % period)
                padding = torch.zeros([x_embed.shape[0], length, x_embed.shape[2], x_embed.shape[-1]]).to(x_embed.device)
                out = torch.cat([x_embed, padding], dim=1)
            else:
                length = self.d_model
                out = x_embed

            new_x_inner = out.unfold(dimension=1, size=period,step=period)  # [b x patch_num x nvar x dim x patch_len]
            b,period_num,nvar,dim,paroid_len = new_x_inner.size()
            new_x_inner = self.emb_linear(new_x_inner.permute(0,1,2,4,3)) # b,period_num,nvar,paroid_len,dim
            new_x_inner = new_x_inner.permute(0,2,3,1,4).contiguous() # b,nvar,paroid_len,period_num,dim

            W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=period_num,
                                        d_model=dim)
            W_pos = W_pos.to(new_x_inner.device)
            new_x_inner = self.dropout(new_x_inner + W_pos)

            # q_b, q_nvar, q_dim, q_period_num, q_paroid_len
            new_x_inner = new_x_inner.permute(0,1,4,3,2) # b,nvar,dim,period_num,paroid_len
            # new_x_inner = torch.reshape(new_x_inner, (b *nvar, dim,period_num,paroid_len))
            # new_x_inner = torch.reshape(new_x_inner, (b *nvar, dim,period_num*paroid_len))

            # Positional encoding

            out,att = self.period_attention(Q=new_x_inner,K=new_x_inner,V=new_x_inner)
            #  b*nvar,period_num,paroid_len * dim
            out_bs,out_dim,out_seq = out.shape
            out = torch.reshape(out, (b, nvar,out_dim,out_seq)).contiguous()

            out = out.permute(0,3,1,2).contiguous() # b, out_seq, nvar, out_dim
            res.append(out)
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = self.un_embed_dim(res.permute(0,1,2,4,3)).contiguous().squeeze(-1)
        res = torch.sum(res * period_weight, -1)  # 得到时间频率重要的部分的权重

        # # residual connection
        # res = self.de_embed(res.permute(0,2,1)).permute(0,2,1).contiguous()

        # fourier

        x_fourier_embed_enc = self.fourier_zero(x_fourier_embed,time_step=self.configs.time_step,point_frequence=4096) # mse:0.17436252534389496, mae:0.25599
        x_fourier_embed_enc_list = []
        for i, x_f in zip(range(len(x_fourier_embed_enc)), x_fourier_embed_enc, ):
            B_f, T_f, N_f = x_f.size()
            x_f = self.normalize_layers[i](x_f, 'norm')
            if self.channel_independence == 1:
                x_f = x_f.permute(0, 2, 1).contiguous().reshape(B_f * N_f, T_f, 1)
            x_fourier_embed_enc_list.append(x_f)
        enc_out_list = x_fourier_embed_enc_list
        x_fourier_embed_dec_list = self.pre_enc(x_fourier_embed_enc_list)
        dec_out_list = self.local_global_enc(B, enc_out_list, x_fourier_embed_dec_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        result = self.normalize_layers[0](dec_out, 'denorm')
        # result = self.de_embed(result.permute(0,2,1)).permute(0,2,1).contiguous()

        if self.configs.use_atten == 1:
            res_att = res
        else:
            res_att = 0

        if self.configs.use_fourier_att == 1:
            res_fourier = result
        else:
            res_fourier = 0
        res = res_att + res_fourier

        # residual connection
        res = self.de_embed(res.permute(0, 2, 1)).permute(0, 2, 1).contiguous()



        out = res + new_x # 32 * 96 * 7

        ##FFN
        out = self.dropout(out)
        out = self.ff(out.permute(0,2,1)).permute(0,2,1).contiguous() + out
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = patch_size_list
        # self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.layers = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        for num in range(self.layer_nums):
            self.layers.append(
                DynoFormerLayer(self.configs)
            )
        self.projections = nn.Sequential(
            nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        )



    def forward(self, x,batch_x_mark, dec_inp, batch_y_mark):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.projections(out.permute(0,2,1)).permute(0,2,1).contiguous()
        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        return out
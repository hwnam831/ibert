import torch
import torch.nn as nn
from Encoder import TFEncoder, CNNEncoder
from Decoder import TFDecoder
from torch.nn import functional as F



class TfS2S(nn.Module):
    def __init__(self, model_size=512, maxlen=128):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Embedding(16, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.encoder = TFEncoder(model_size, nhead=2, num_layers=3)
        self.decoder = TFDecoder(model_size, 16)
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, target):
        input2 = input.permute(1,0)
        target2 = target.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape)
        tpos = torch.arange(target2.size(0), device=target.device)[:,None].expand(target2.shape)

        src = self.embedding(input2) + self.posembed(ipos)
        tgt = self.embedding(target2)+ self.posembed(tpos)
        memory = self.encoder(src)
        return self.decoder(tgt, memory).permute(1,2,0)

class TfAE(nn.Module):
    def __init__(self, model_size=512, maxlen=128):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Linear(16, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=2)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, num_layers=6, norm=self.norm)
        self.fc = nn.Linear(model_size, 16)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        src = self.embedding(input2) + self.posembed(ipos)
        out = self.tfmodel(src)
        return self.fc(out).permute(1,2,0)

class CNNAE(nn.Module):
    def __init__(self, model_size=512):
        super().__init__()
        self.model_size=model_size
        self.embedding = nn.Conv1d(16, model_size, 3, padding=1)
        self.norm = nn.BatchNorm1d(model_size)
        self.encoder = CNNEncoder(model_size)
        self.fc = nn.Linear(model_size, 16)
    
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        embed = self.norm(self.embedding(input.permute(0,2,1)))
        out = self.encoder(embed.permute(2,0,1))
        return self.fc(out).permute(1,2,0)


class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__())
        self.output_atten = config.output_attentions

        if config.d_model % config.n_head != 0:
            raise ValueError("Hidden size need to be multiple of number of attention. correct d_model and n_head values")
    
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1/(config.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bais = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bais = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bais = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embd = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)


        def rel_shift(x, klen=-1):
            x_size = x.shape

            x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
            x = x[1:, ...]
            x = x.reshape(x_size[0], x_size[1]-1, x_size[2], x_size[3])
            x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype = torch.long))

            return x

        def rel_shift_bnji(x, klen=-1):
            x_size = x.shape

            x = x.reshape(x_size[0], x_size[0], x_size[3], x_size[2])
            x = x[:,:,1:,:]
            x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3]-1)
            x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype = torch.long))

            return x

        //TODO : Remove seg_mat
        def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):

            ac = torch.einsum("ibnd, jbnd->bnji", q_head+self.r_w_bais, k_head_h)
            bd = torch.einsum("ibnd, jbnd->bnij", q_head+self.r_r_bais, k_head_r)
            bd = self.rel_shift_bnji(bd, klen = ac.shape[3])

            if seg_mat is None:
                ef = 0 
            else:
                ef = torch.einsum("ibnd, snd->ibns", q_head + self.r_s_bais, self.seg_embd)
                ef = torch.einsum("ijbs, ibns->bnij", seg_mat, ef)

            attn_score = (ac+bd+ef) * self.scale

            if attn_mask is not None:
                if attn_mask.dtype == torch.float16:
                    attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
                else:
                    attn_score = attn_score - 1e30  * torch.einsum("ijbn->bnij", attn_mask)
            
            attn_prob = F.softmax(attn_score, dim=3)
            attn_prob = self.dropout(attn_prob)

            if head_mask is not None:
                attn_prob = attn_prob * torch.einsum("ijbn->bnij", head_mask)

            attn_vec = torch.einsum("bnij, jbnd->ibnd", attn_prob, v_head_h)

            if self.output_attentions:
                return attn_vec, torch.einsum("bnij->ijbn", attn_prob)
            
            return attn_vec



        def post_attention(self, h , attn_vec, residual = True):

            attn_out = torch.einsum("ibnf, hnd->ibh", attn_vec, self.o)
            attn_out = self.dropout(attn_out)

            if residual:
                attn_out = attn_out +h 

            return self.layer_norm(attn_out)

        def forward(self, h, g, attn_mask_h, attn_mask_g, r, seq_mat, mems=None, target_mapping=None, head_mask=None):

            if g is not None:
                "This is two stream attention with relative positional encoding"

                if mems is not None and mems.dim() >1:
                    cat = torch.cat([mems,h], dim=0)
                else:
                    cat = h 
                

                k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
                v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
                k_head_r = torch.einsum("ibh,hnd->ibnd", cat, self.r)
                q_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.q)

                attn_vec_h = self.rel_attn_core(q_head_h,
                                                k_head_h,
                                                v_head_h,
                                                k_head_r,
                                                seq_mat=seq_mat,
                                                attn_mask = attn_mask_h,
                                                head_mask = head_mask
                                                )
                
                if self.output_attentions: 
                    attn_vec_h, attn_prob_h = attn_vec_h
                
                output_h = self.post_attention(h, attn_vec_h)


                q_head_g = torch.einsum("ibh, hnd->ibnd", g, self.q)

                if target_mapping is not None:
                    q_head_g = torch.einsum("mbnd, m1b->1bnd", q_head_g, target_mapping)

                    attn_vec_g = self.rel_attn_core(q_head_q,
                                                    k_head_h,
                                                    v_head_h,
                                                    k_head_r,
                                                    seq_mat=seq_mat,
                                                    attn_mask = attn_mask_h,
                                                    head_mask = head_mask
                                                    )

                    if self.output_attentions: 
                        attn_vec_g, attn_prob_g = attn_vec_g

                    attn_vec_g = torch.einsum("1bnd, m1b->mbnd", attn_vec_g, target_mapping)
                else:
                    attn_vec_g = self.rel_attn_core(q_head_g,
                                                    k_head_h,
                                                    v_head_h,
                                                    k_head_r,
                                                    seq_mat=seq_mat,
                                                    attn_mask = attn_mask_h,
                                                    head_mask = head_mask
                                                    )

                    if self.output_attentions: 
                        attn_vec_g, attn_prob_g = attn_vec_g


                output_g = self.post_attention(g, attn_vec_g)

                if self.output_attentions: 
                    attn_prob = attn_prob_h, attn_vec_g
            
            
            else: 
                "Multihead attention with relative positional encoding"

                if mems is not None and mems.dim() >1:
                    cat = torch.cat([mems,h], dim=0)
                else:
                    cat = h 

                q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
                k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
                v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
                k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)

                attn_vec = self.rel_attn_core(q_head_h,
                                              k_head_h,
                                              v_head_h,
                                              k_head_r,
                                              seq_mat=seq_mat,
                                              attn_mask = attn_mask_h,
                                              head_mask = head_mask
                                              )

                if self.output_attentions: 
                    attn_vec, attn_prob = attn_vec

                output_h = self.post_attention(h, attn_vec)
                output_g = None
            


            outputs = (output_h, output_g)

            if self.output_attentions:
                outputs = outputs + (attn_prob,)                
            
            return outputs

class XLNetFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model, eps = config.layer_norm_eps)
        self.layer_1    = nn.Linear(config.d_model, config.d_inner)
        self.layer_2    = nn.Linear(config.inner, config.d_model)
        self.dropout    = nn.Dropout(config.dropout)
        self.activation_function = config.ff_activation

    def forward(self, inp):
        output = self.layer_1(inp)
        output = self.activation_function(output)
        output = self.dropout(output)

        output = self.layer_2(inp)
        output = self.activation_function(output)
        output = self.dropout(output)

        return layer_norm(output + inp)


class XLNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFF(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        output = self.rel_attn(output_h, 
                               output_g, 
                               attn_mask_h, 
                               attn_mask_g,
                               r, 
                               seq_mat,
                               mems=mems, 
                               target_mapping=target_mapping, 
                               head_mask=head_mask
                               )
        output_h, output_g = output[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        output = (output_h, output_g) + output[2:]
        return output

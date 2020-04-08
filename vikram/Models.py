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
    # def __init__(self, d_model=512, n_head=3, d_head=3, d_inner=3, layer_norm_eps, dropout, ff_activation="relu"):#, output_attentions=False):
        super().__init__())

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

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
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

        # //TODO : Remove seg_mat
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

            # if self.output_attentions:
            #     return attn_vec, torch.einsum("bnij->ijbn", attn_prob)
            
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
                
                # if self.output_attentions: 
                #     attn_vec_h, attn_prob_h = attn_vec_h
                
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

                    # if self.output_attentions: 
                    #     attn_vec_g, attn_prob_g = attn_vec_g

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

                    # if self.output_attentions: 
                    #     attn_vec_g, attn_prob_g = attn_vec_g


                output_g = self.post_attention(g, attn_vec_g)

                # if self.output_attentions: 
                #     attn_prob = attn_prob_h, attn_vec_g
            
            
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

                # if self.output_attentions: 
                #     attn_vec, attn_prob = attn_vec

                output_h = self.post_attention(h, attn_vec)
                output_g = None
            


            outputs = (output_h, output_g)

            # if self.output_attentions:
            #     outputs = outputs + (attn_prob,)                
            
            return outputs

class XLNetFF(nn.Module):
    def __init__(self, config):
    # def __init__(self, d_model=512, n_head=3, d_head=3, d_inner=3, layer_norm_eps, dropout, ff_activation="relu"):#, output_attentions=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model, eps =config.layer_norm_eps)
        self.layer_1    = nn.Linear(config.d_model, config.d_inner)
        self.layer_2    = nn.Linear(config.d_inner, config.d_model)
        self.dropout    = nn.Dropout(config.dropout)
        # self.activation_function = config.ff_activation

    def forward(self, inp):
        output = self.layer_1(inp)
        # output = self.activation_function(output)
        output = nn.GELU(output)
        output = self.dropout(output)

        output = self.layer_2(inp)
        # output = self.activation_function(output)
        output = nn.GELU(output)
        output = self.dropout(output)

        return layer_norm(output + inp)


class XLNetLayer(nn.Module):
    def __init__(self, config):
#    def __init__(self, d_model=512, n_head=3, d_head=3, d_inner=3, layer_norm_eps, dropout, ff_activation="relu"):#, output_attentions=False):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFF(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat=None, mems=None, target_mapping=None, head_mask=None):
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




class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_past = config.output_past

        self.mem_len    = config.mem_len
        self.reuse_len  = config.reuse_len
        self.d_model    = config.d_model
        self.same_length= config.same_length
        self.attn_type  = config.attn_type
        self.bi_data    = config.bi_data
        self.clamp_len  = config.clamp_len
        self.n_layer    = config.n_layer

        self.mask_emb = nn.Parameter(torch.FloatTensor(1,1,config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

    
    def create_mask(self, qlen, mlen):
        """
        args: qlen - sequence length
              mlen - mask length

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """

        attn_mask       = torch.ones([qlen, qlen])
        mask_up         = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad   = torch.zeros([qlen,qlen])
        ret             = torch.cat([attn_mask_pad, mask_up], dim=1)

        if self.same_length :
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret     = torch.cat([ret[:, :qlen,mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(next(self.parameters()))
        return ret


    def cache_mem:
        if self.reuse_len is not None and self.reuse_len >0:
            curr_out = curr_out[:self.reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-self.mem_len:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]

        return new_mem.detach()

    
    
    def positional_embedding(pos_seq, inv_freq, bsz=None):

        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz= None):

        freq_seq = torch.arrange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1/torch.pow(10000, (freq_seq/self.d_model))

        if self.attn_type == "bi":
            beg, end = klen, -qlen
        else if self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError("Uknown attn_type")

        
        if self.bi_data:
            fwd_pos_seq = torch.arrange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arrange(-beg, end, -1.0, dtype=torch.float)

            if self.clamp_len  >0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if self.bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz //2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz //2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb])

        else:
            fwd_pos_seq = torch.arrange(beg, end, -1.0)
            if self.clamp_len >0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameter()))
        return pos_emb


    def forward(self, 
                input_ids=None, 
                attention_mask= None, 
                mems= None, 
                perm_mask=None, 
                target_mapping=None,
                token_type_ids=None,
                input_mask=None, 
                head_mask=None, 
                inputs_embeds=None,
                ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Input ids and embds cannot be set")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0,1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0,1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("Either inputid or embds is needed")

        
        token_type_ids = token_type_ids.transpose(0,1).contiguous() if token_type_ids is not None else None 
        input_mask = input_mask.transpose(0,1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0,1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.transpose(0,1).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.transpose(0,1).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        #TODO
        dtype_float = next(self.parameter()).dtype
        device = next(self.parameter()).device


        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[]:,:,None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError("Need to specify attn_mask type")


        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None


        if data_mask is not None:
            if mlen >0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:,:,:, None]
            else:
                attn_mask += data_mask[:,:,:, None]

        if attn_mask is not None:
            attn_mask = (attn_mask>0).to(dtype_float)

        if attn_mask  is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            
            if mlen >0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask+non_tgt_mask[:,:,None,None])>0).to(attn_mask)

        else:
            non_tgt_mask = None
        
        #No word embeddings
        output_g = None
        
        if token_type_ids is not None:
            if mlen > 0: 
                mem_pad = torch.zeros([mlen, bsz], dtype = torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids
            
            seg_mat =  (token_type_ids[:,None] != cat_ids[None,:]).long()
            seg_mat = F.one_hot(seg_mat,num_classes=2).to(dtype_float)
        else: 
            seg_mat = None
        

        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif head_mask.dim() == 2:
                head_mask = head_mask.expand(self.n_layer, -1,-1,-1,-1)
            head_mask = head_mask.to(dtype=next(self.parameter()).dtype)
        else :
            head_mask = [None] * self.n_layer
        

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)
        
        attentions = []
        hidden_states = []

        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len >0 and self.output_past :
                new_mems = new_mems+ (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)


            outputs = layer_module(
                            output_h,
                            output_g,
                            attn_mask_h=non_tgt_mask,
                            attn_mask_g=attn_mask,
                            r=pos_emb,
                            seg_mat=seg_mat,
                            mems=mems[i],
                            target_mapping=target_mapping,
                            head_mask=head_mask[i],
                         )
            output_h, output_g = output[:2]
            if self.output_attentions:
                attentions.append(output[2])
            

        if self.output_hidden_states:
            hidden_states.appen(output_h, output_g) if output_g is None else output_h
        
        output = self.dropout(output_g if output_g is not None else output_h)
    
        output = (output.permute(1,0,2).contiguous(),)

        if self.mem_len is not None and self.mem_len >0 and self.output_past:
            outputs = outputs_(new_mems,)
        
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1,0,2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1,0,2).contiguous() for hs in hidden_states)
            
            outputs = outputs + (hidden_states,)

        if self.output_attentions:
            if target_mapping is not None:
                attentions = tuple(tuple(att_stream.permute(2,3,0,1).contiguous() for att_stream in t) for t in attentions)
            else: 
                attentions = tuple(t.permute(2,3,0,1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)
        
        return outupts









class XLNetConfig(object):
    output_attentions=False
    output_hidden_states=False
    # hidden_size=256
    # num_hidden_layers=6
    # num_attention_heads=2
    # intermediate_size=2048
    # hidden_dropout_prob=0.1
    # attention_probs_dropout_prob=0.1
    # max_position_embeddings=128
    # layer_norm_eps=1e-12
    d_model=512
    n_head=3
    d_head=3
    d_inner=3
    layer_norm_eps
    dropout=0.1
    ff_activation="gelu"

from transformers import AutoModel, AutoConfig
import torch
from torch import nn
from opt_einsum import contract
import torch.nn.functional as F
from model.mlp import MLP


# class Biaffine(nn.Module):
#     def __init__(self, hidden_dim, num_class, dropout=0.2):
#         super(Biaffine, self).__init__()
#         self.hidden_dim = hidden_dim
#         self._hidden_dim = hidden_dim // 2
#         self.num_class = num_class
#         self.dropout = dropout
#         self.linear_h = nn.Sequential(
#             nn.Linear(self.hidden_dim, self._hidden_dim),
#             nn.Dropout(self.dropout),
#             nn.ReLU(),
#             nn.Linear(self._hidden_dim, self._hidden_dim),
#             nn.Dropout(self.dropout),
#             nn.ReLU()
#         )
#         self.linear_t = nn.Sequential(
#             nn.Linear(self.hidden_dim, self._hidden_dim),
#             nn.Dropout(self.dropout),
#             nn.ReLU(),
#             nn.Linear(self._hidden_dim, self._hidden_dim),
#             nn.Dropout(self.dropout),
#             nn.ReLU()
#         )
#         self.tri = nn.Parameter(torch.randn(self.num_class, self._hidden_dim, self._hidden_dim))
#         self.bi0 = nn.Parameter(torch.randn(self.num_class, self._hidden_dim))
#         self.bi1 = nn.Parameter(torch.randn(self.num_class, self._hidden_dim))
#         self.uni = nn.Parameter(torch.randn(self.num_class))
        
#     def forward_self(self, memory):
#         '''
#         memory: B * S * H
#         hs: B * (1) * Q(S) * H
#         output: B * Q(S) * S * C
#         '''
#         bsz, seq, hidden = memory.size(0), memory.size(1), memory.size(-1)
        
#         head = self.linear_h(memory)
#         tail = self.linear_t(memory)
#         tri_score = contract('bso,bqp,cop->bsqc', head, tail, self.tri)
#         bi_score = contract('bso,co->bsc', head, self.bi0).unsqueeze(2) + \
#                    contract('bqp,cp->bqc', tail, self.bi1).unsqueeze(1)
#         uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#         return tri_score + bi_score + uni_score
 
#     def forward_query(self, memory, head, tail):
#         '''
#         memory: B * S * H
#         head: B * (L) * Q * S
#         tail: B * (L) * Q * S
#         output: B * (L) * Q * C
#         '''
#         bsz, seq = memory.size(0), memory.size(1)
#         query = head.size(-2)
        
#         head = head.reshape(bsz, -1, seq)
#         tail = tail.reshape(bsz, -1, seq)
        
#         head_prob = nn.Softmax(dim=-1)(head)
#         tail_prob = nn.Softmax(dim=-1)(tail)
        
#         head_rep = torch.bmm(head_prob, self.linear_h(memory)) # B * Q * H // 2
#         tail_rep = torch.bmm(tail_prob, self.linear_t(memory))
        
#         tri_score = contract('bqo,bqp,cop->bqc', head_rep, tail_rep, self.tri)
#         bi_score = contract('bqo,co->bqc', head_rep, self.bi0) + \
#                    contract('bqp,cp->bqc', tail_rep, self.bi1)
#         uni_score = self.uni.unsqueeze(0).unsqueeze(0)
#         return (tri_score + bi_score + uni_score).reshape(bsz, -1, query, self.num_class)
    
#     def forward(self, memory, head=None, tail=None):
#         if head is None and tail is None:
#             return self.forward_self(memory)
#         return self.forward_query(memory, head, tail)

# Adopt From TreeCRF
class BiAffineBase(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super().__init__()

        self.parse_proj = nn.Parameter(
            torch.randn(num_class, hidden_dim, hidden_dim))
        self.offset_proj = nn.Parameter(
            torch.randn(hidden_dim, num_class))
        self.offset = nn.Parameter(torch.randn(num_class))
        return

    def forward(self, sent_states):
        label_size = self.parse_proj.size(0)
        batch_size = sent_states.size(0)
        max_len = sent_states.size(1)
        hidden_size = sent_states.size(2)
        sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
        sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
        parse_proj = self.parse_proj.view(1, label_size, hidden_size, hidden_size)

        # project to CRF potentials

        # binear part
        # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
        proj = torch.matmul(sent_states, parse_proj)
        # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
        log_potentials = torch.matmul(proj, sent_states_)
        # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
        log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
        # [[batch, len * len, label] -> [[batch, len, len, label]
        log_potentials_0 = log_potentials.view(batch_size, max_len, max_len, label_size)

        # local offset
        sent_states_sum_0 = sent_states.view(batch_size, max_len, 1, hidden_size)
        sent_states_sum_1 = sent_states.view(batch_size, 1, max_len, hidden_size)
        # [batch, len, 1, hidden] + [batch, 1, len, hidden] -> [batch, len, len, hidden]
        sent_states_sum = (sent_states_sum_0 + sent_states_sum_1).view(batch_size, -1, hidden_size)
        offset_proj = self.offset_proj.view([1, hidden_size, -1])
        # [batch, len * len, hidden] * [1, hidden, label] -> [batch, len * len, label]
        log_potentials_1 = torch.matmul(sent_states_sum, offset_proj)
        log_potentials_1 = log_potentials_1.view(batch_size, max_len, max_len, label_size)

        offset = self.offset.view(1, 1, 1, label_size)
        log_potentials = log_potentials_0 + log_potentials_1 + offset
        return log_potentials


class Biaffine(nn.Module):
    def __init__(self, hidden_dim, num_class, dropout=0.2):
        super().__init__()

        _hidden_dim = hidden_dim // 2
        self.biaffine = BiAffineBase(_hidden_dim, num_class)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, _hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(_hidden_dim, _hidden_dim),
            nn.Dropout(dropout)
        )
        return

    def forward(self, sent_states):
        sent_states = self.linear(sent_states)
        log_potentials = self.biaffine(sent_states)
        return log_potentials
        

class TypeAttention(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(TypeAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        
        self.W = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.U = nn.Linear(self.hidden_dim, self.num_class)
        self.V = nn.Linear(self.hidden_dim, self.num_class)
        
    def forward_query(self, memory, head, tail):
        bsz, seq = memory.size(0), memory.size(1)
        query = head.size(-2)
        
        head = head.reshape(bsz, -1, seq)
        tail = tail.reshape(bsz, -1, seq)
        
        z = nn.ReLU()(self.W(memory)) # B * S * H
        score = self.U(z).unsqueeze(1).repeat(1, head.size(1), 1, 1) # B * Q * S * C
        
        idx = torch.arange(memory.size(1)).unsqueeze(0).unsqueeze(0).to(memory.device) # B * Q * S
        head_idx = torch.max(head, dim=-1)[1] # B * Q
        tail_idx = torch.max(tail.masked_fill(idx < head_idx.unsqueeze(-1), value=-1e6), dim=-1)[1]
        mask_idx = torch.bitwise_or(idx < head_idx.unsqueeze(-1), idx > tail_idx.unsqueeze(-1))
        
        score = score.masked_fill(mask_idx.unsqueeze(-1).expand_as(score), value=-1e6)
        alpha = nn.Softmax(dim=-2)(score) # B * Q * S * C
        type_span_h = torch.matmul(z.unsqueeze(1).permute(0,1,3,2), alpha) # B * Q * H * C
        score = contract('bqhc,ch->bqc', type_span_h, self.V.weight) + self.V.bias
        return score.reshape(bsz, -1, query, self.num_class)
    
    def forward_self(self, memory):
        '''
        memory: B * S * H
        hs: B * (1) * Q(S) * H
        output: B * Q(S) * S * C
        '''
        z = torch.relu(self.W(memory)) # B * L * H
        score = self.U.weight.matmul(z.transpose(1,2)) # B * R * L
        max_score = torch.max(score, dim=-1, keepdim=True).values # B * R * L
        exp_score = torch.exp(score - max_score) # B * R * L
        cum_exp_score = torch.cumsum(exp_score, dim=2).permute(0,2,1) # B * L * R
        cum_cum_exp_score = torch.cat((cum_exp_score.unsqueeze(1), (cum_exp_score.unsqueeze(1) - cum_exp_score.unsqueeze(2))[:,:-1]), dim=1)
        cum_cum_exp_score.masked_fill_(cum_cum_exp_score <= 1e-6, 100.)
        # B L L R
        
        m = memory.unsqueeze(2) * exp_score.permute(0,2,1).unsqueeze(-1) # B * L * R * H
        cum_m = torch.cumsum(m, dim=1)
        cum_cum_m = torch.cat((cum_m.unsqueeze(1), (cum_m.unsqueeze(1) - cum_m.unsqueeze(2))[:,:-1]), dim=1)
        # B L L R H
        
        type_span_h = torch.relu(cum_cum_m / cum_cum_exp_score.unsqueeze(-1)) # B L L R H
        score = contract('bijrh,rh->bijr', type_span_h, self.V.weight) + self.V.bias
        return score
    
    def forward(self, memory, head=None, tail=None):
        if head is None and tail is None:
            return self.forward_self(memory)
        return self.forward_query(memory, head, tail)
    

class TriAffine(nn.Module):
    def __init__(self, n_in, num_class, bias_x=True, bias_y=True, scale="none", init_std=0.01):
        super().__init__()

        self.n_in = n_in
        self.num_class = num_class
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in + bias_x,
                                                n_in,
                                                n_in + bias_y,
                                                num_class))
        self.init_std = init_std
        self.scale = scale
        self.calculate_scale_factor()
        self.reset_parameters()
        
    def calculate_scale_factor(self):
        if self.scale == "none":
            self.scale_factor = 1
        elif self.scale == "sqrt":
            self.scale_factor = self.n_in ** (-0.5)
        elif self.scale.find("tri") >= 0:
            self.scale_factor = self.n_in ** (-1.5) * self.init_std ** (-1)

    def extra_repr(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        s += f", num_class={self.num_class}"
        return s

    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        nn.init.normal_(self.weight, std=self.init_std)

    def forward(self, x, y, z):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, n_in]
            y (torch.Tensor): [batch_size, seq_len, n_in]
            z (torch.Tensor): [batch_size, seq_len, n_in]
        Returns:
            s (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        #w = contract('bzk,ikjr->bzijr', z, self.weight) # bsz * seq * h * h * class
        #s = contract('bxi,bzijr,byj->bzxyr', x, w, y) # bsz * seq * seq * seq * class
        #s = contract('bxi,bzk,ikjr,byj->bzxyr', x, z, self.weight, y)
        #s = contract('bxi,bzk,ikjr,byj->bxzyr', x, z, self.weight, y)
        s = contract('bxi,bzk,ikjr,byj->bxyzr', x, z, self.weight, y)

        if self.num_class == 1:
            return s.squeeze(-1)
        
        if hasattr(self, 'scale_factor'):
            s = s * self.scale_factor
        return s
    
    def forward_query(self, x, y, z):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
            
        s = contract('bxi,bzk,ikjr,bxj->bzxr', x, z, self.weight, y)
            
        if self.num_class == 1:
            return s.squeeze(-1)
        
        if hasattr(self, 'scale_factor'):
            s = s * self.scale_factor
        return s
    
class TriAffinePos(TriAffine):
    def __init__(self, n_in, num_class, bias_x=True, bias_y=True, scale="none",
                 init_std=0.01, k=64):
        super(TriAffinePos, self).__init__(n_in,num_class,bias_x,bias_y,scale,init_std)
        
        # self.init_bi_std = init_bi_std
        # self.weight_xz = nn.Parameter(torch.Tensor(n_in + bias_x, n_in, num_class))
        # self.weight_yz = nn.Parameter(torch.Tensor(n_in + bias_y, n_in, num_class))
        # self.weight_xy = nn.Parameter(torch.Tensor(n_in, n_in, num_class))
        # self.reset_bi_parameters()
        
        self.k = k
        self.pos_xz = nn.Embedding(2 * k + 1, n_in + bias_y)
        self.pos_yz = nn.Embedding(2 * k + 1, n_in + bias_x)
        self.pos_xy = nn.Embedding(2 * k + 1, n_in)
        
    # def reset_bi_parameters(self):
    #     nn.init.normal_(self.weight_xz, std=self.init_bi_std)
    #     nn.init.normal_(self.weight_yz, std=self.init_bi_std)
    #     nn.init.normal_(self.weight_xy, std=self.init_bi_std)
        
    def move(self, dis):
        return torch.clamp(dis + self.k, 0, 2 * self.k).long()
        
    def forward(self, x, y, z, idx_x, idx_y, idx_z0, idx_z1=None):
        """
        x: B * seq * n_in
        y: B * seq * n_in
        z: B * seq * n_in
        idx_x, idx_y, idx_z0, idx_z1: B * seq
        """
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
            
        s = contract('bxi,bzk,ikjr,byj->bxyzr', x, z, self.weight, y)
            
        dis_xz = idx_z0.unsqueeze(1) - idx_x.unsqueeze(2)
        emb_xz = self.pos_xz(self.move(dis_xz)) # b * seq_x * seq_z * n_in
        s_xz = contract('bxi,bzk,ikjr,bxzj->bxzr', x, z, self.weight, emb_xz)
        
        dis_xy = idx_y.unsqueeze(1) - idx_x.unsqueeze(2)
        emb_xy = self.pos_xy(self.move(dis_xy))
        s_xy = contract('bxi,byj,ikjr,bxyk->bxyr', x, y, self.weight, emb_xy)
        
        if idx_z1 is not None:
            dis_yz = idx_y.unsqueeze(2) - idx_z1.unsqueeze(1)
        else:
            dis_yz = idx_y.unsqueeze(2) - idx_z0.unsqueeze(1)
        emb_yz = self.pos_yz(self.move(dis_yz))
        s_yz = contract('byj,bzk,ikjr,byzi->byzr', y, z, self.weight, emb_yz)
        
        s += s_xz.unsqueeze(2) + s_xy.unsqueeze(3) + s_yz.unsqueeze(1)

        if self.num_class == 1:
            return s.squeeze(-1)
        
        if hasattr(self, 'scale_factor'):
            s = s * self.scale_factor
        return s
        
    
class TriAttention(nn.Module):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True,
                 reduce_last=False, dropout=0.0):
        super(TriAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.parser = TriAffine(self._hidden_dim, self.num_class)
        self.mask = mask
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.V = nn.Linear(self.hidden_dim, self.num_class)
        
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        score = self.parser(head, tail, mid) # b * seq * seq * seq * type
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        type_span_h = torch.relu(contract('bkh,bijkr->bijrh', memory, alpha))
        score = contract('bijrh,rh->bijr', type_span_h, self.V.weight) + self.V.bias
        return score
    
    def forward_query(self, memory, head, tail):
        '''
        memory: B * S * H
        head: B * (L) * Q * S
        tail: B * (L) * Q * S
        output: B * (L) * Q * C
        '''
        bsz, seq = memory.size(0), memory.size(1)
        query = head.size(-2)
        
        head = head.reshape(bsz, -1, seq)
        tail = tail.reshape(bsz, -1, seq)
        head_prob = nn.Softmax(dim=-1)(head)
        tail_prob = nn.Softmax(dim=-1)(tail)
        
        head_rep = torch.bmm(head_prob, self.linear_h(memory)) # B * Q * H1
        tail_rep = torch.bmm(tail_prob, self.linear_t(memory))
        mid_rep = self.linear_m(memory)
        
        score = self.parser.forward_query(head_rep, tail_rep, mid_rep).permute(0,2,1,3) # B * query * seq * type
        seq_t = torch.arange(seq).to(memory.device).unsqueeze(0).unsqueeze(0) # B * query * seq
        
        head_idx = torch.max(head_prob, dim=-1)[1] # B * query
        tail_idx = torch.max(tail_prob, dim=-1)[1]
        
        if self.mask:
            score.masked_fill_(torch.bitwise_or(seq_t < head_idx.unsqueeze(-1), seq_t > tail_idx.unsqueeze(-1)).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        type_span_h = torch.relu(contract('bsh,bqsr->bqrh', memory, alpha))
        final_score = contract('bqrh,rh->bqr', type_span_h, self.V.weight) + self.V.bias
        return final_score.reshape(bsz, -1, query, self.num_class)
    
    def forward(self, memory, head=None, tail=None):
        if head is None and tail is None:
            return self.forward_self(memory)
        return self.forward_query(memory, head, tail)
        
class TriAffineParser(nn.Module):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False,
                 rel_pos_attn=False, rel_pos=False, k=64):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
            
        self.rel_pos_attn = rel_pos_attn
        self.rel_pos = rel_pos
        if self.rel_pos_attn:
            self.parser0 = TriAffinePos(self._hidden_dim, self.num_class, scale=scale, init_std=init_std, k=k)
        else:
            self.parser0 = TriAffine(self._hidden_dim, self.num_class, scale=scale, init_std=init_std)
        if self.rel_pos:
            self.parser1 = TriAffinePos(self._hidden_dim, self.num_class, scale="none", init_std=init_std, k=k)
        else:
            self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_h = nn.LayerNorm(self._hidden_dim)
            self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
        
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head = self.norm_h(head)
            tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        seq_t = torch.arange(seq).to(memory.device)
        """
        if not self.rel_pos_attn:
            score = self.parser0(head, tail, mid) # b * seq * seq * seq * type
        else:
            seq_tb = seq_t.unsqueeze(0)
            score = self.parser0(head, tail, mid, seq_tb, seq_tb, seq_tb)
        """
        if hasattr(self, 'rel_pos_attn') and self.rel_pos_attn:
            seq_tb = seq_t.unsqueeze(0)
            score = self.parser0(head, tail, mid, seq_tb, seq_tb, seq_tb)
        else:
            score = self.parser0(head, tail, mid) # b * seq * seq * seq * type


        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        head1 = self.linear_h1(memory)
        tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head1 = self.norm_h1(head1)
            tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        if hasattr(self, 'rel_pos') and self.rel_pos:
            score1 = self.parser1(head1, tail1, mid1, seq_tb, seq_tb, seq_tb)
        else:
            score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1
    
    def forward_query(self, memory, head, tail):
        raise NotImplementedError
        # bsz, seq = memory.size(0), memory.size(1)
        # query = head.size(-2)
        
        # head = head.reshape(bsz, -1, seq)
        # tail = tail.reshape(bsz, -1, seq)
        # head_prob = nn.Softmax(dim=-1)(head)
        # tail_prob = nn.Softmax(dim=-1)(tail)
        
        # head_rep = torch.bmm(head_prob, self.linear_h(memory)) # B * Q * H1
        # tail_rep = torch.bmm(tail_prob, self.linear_t(memory))
        # mid_rep = self.linear_m(memory)
        # score = self.parser0.forward_query(head_rep, tail_rep, mid_rep).permute(0,2,1,3) # B * query * seq * type
        
        # seq_t = torch.arange(seq).to(memory.device).unsqueeze(0).unsqueeze(0) # B * query * seq
        
        # if self.mask:
        #     head_idx = torch.max(head_prob, dim=-1)[1] # B * query
        #     tail_idx = torch.max(tail_prob, dim=-1)[1]
        #     score.masked_fill_(torch.bitwise_or(seq_t < head_idx.unsqueeze(-1), seq_t > tail_idx.unsqueeze(-1)).unsqueeze(-1), -1e6)
        # alpha = F.softmax(score, dim=-2)
        
        # head1 = torch.bmm(head_prob, self.linear_h1(memory))
        # tail1 = torch.bmm(tail_prob, self.linear_t1(memory))
        # mid1 = self.linear_m1(memory)
        # score1 = self.parser1.forward_query(head1, tail1, mid1).permute(0,2,1,3) # B * query * seq * type
        # uni_score = self.uni.unsqueeze(0).unsqueeze(0)
        # final_score = contract('bqsr,bqsr->bqr', alpha, score1) + uni_score
        # final_score = final_score.reshape(bsz, -1, query, self.num_class)
        
        # if not self.return_attention:
        #     return final_score
        # return final_score, alpha, head1, tail1, mid1
    
    def forward(self, memory, head=None, tail=None):
        if head is None and tail is None:
            return self.forward_self(memory)
        return self.forward_query(memory, head, tail)
        
# For ablation
class TriAffineParserWithoutLable(TriAffineParser):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
            
        self.parser0 = TriAffine(self._hidden_dim, 1, scale=scale, init_std=init_std)
        self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_h = nn.LayerNorm(self._hidden_dim)
            self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
    
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head = self.norm_h(head)
            tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        score = self.parser0(head, tail, mid).unsqueeze(-1) # b * seq * seq * seq * type
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        head1 = self.linear_h1(memory)
        tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head1 = self.norm_h1(head1)
            tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha.repeat(1,1,1,1,self.num_class), score1) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1
    
class TriAffineParserWithoutBoundary(TriAffineParser):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
            
        # self.parser0 = TriAffine(self._hidden_dim, 1, scale=scale, init_std=init_std)
        self.query0 = nn.Parameter(torch.randn(self.num_class, self._hidden_dim))
        self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            # self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            # self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            # self.norm_h = nn.LayerNorm(self._hidden_dim)
            # self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
    
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        # head = self.linear_h(memory)
        # tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            # head = self.norm_h(head)
            # tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        # score = self.parser0(head, tail, mid).unsqueeze(1) # b * seq * seq * seq * type
        score = contract('bsh,rh->bsr', mid, self.query0).unsqueeze(1).unsqueeze(1).repeat(1,seq,seq,1,1)
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        head1 = self.linear_h1(memory)
        tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head1 = self.norm_h1(head1)
            tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1
    
class TriAffineParserWithoutScorer(TriAffineParser):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
            
        self.parser0 = TriAffine(self._hidden_dim, self.num_class, scale=scale, init_std=init_std)
        self.scorer1 = nn.Linear(self._hidden_dim, self.num_class)
        # self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_h = nn.LayerNorm(self._hidden_dim)
            self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            # self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            # self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
    
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head = self.norm_h(head)
            tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        score = self.parser0(head, tail, mid) # b * seq * seq * seq * type
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        # head1 = self.linear_h1(memory)
        # tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            # head1 = self.norm_h1(head1)
            # tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        score1 = self.scorer1(mid1).unsqueeze(1).unsqueeze(1)
        # score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1.repeat(1,seq,seq,1,1)) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1
    
class TriAffineParserWithoutScorerPlusBoundary(TriAffineParser):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
            
        self.parser0 = TriAffine(self._hidden_dim, self.num_class, scale=scale, init_std=init_std)
        self.scorer1 = nn.Linear(self._hidden_dim, self.num_class)
        self.scorer_h = nn.Linear(self._hidden_dim, self.num_class)
        self.scorer_t = nn.Linear(self._hidden_dim, self.num_class)
        # self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            # self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_h = nn.LayerNorm(self._hidden_dim)
            self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            # self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            # self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
    
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head = self.norm_h(head)
            tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        score = self.parser0(head, tail, mid) # b * seq * seq * seq * type
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        # head1 = self.linear_h1(memory)
        # tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            # head1 = self.norm_h1(head1)
            # tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        score1 = self.scorer1(mid1).unsqueeze(1).unsqueeze(1)
        # score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1.repeat(1,seq,seq,1,1)) + uni_score
        final_score = final_score + self.scorer_h(head).unsqueeze(2) + self.scorer_t(tail).unsqueeze(1)
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1

class LinearTriParser(nn.Module):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0):
        super(LinearTriParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            self.attention_dim = self.hidden_dim
        else:
            self.attention_dim = attention_dim
        self._hidden_dim = self.attention_dim
        self.mask = mask
            
        self.score0_h = nn.Linear(self.attention_dim, self.num_class)
        self.score0_t = nn.Linear(self.attention_dim, self.num_class)
        self.score0_m = nn.Linear(self.attention_dim, self.num_class)
        self.score1_h = nn.Linear(self.attention_dim, self.num_class)
        self.score1_t = nn.Linear(self.attention_dim, self.num_class)
        self.score1_m = nn.Linear(self.attention_dim, self.num_class)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention

    def forward(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        score = self.score0_h(head).unsqueeze(2).unsqueeze(3) + self.score0_t(tail).unsqueeze(1).unsqueeze(3) + self.score0_m(mid).unsqueeze(1).unsqueeze(2)
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        head1 = self.linear_h1(memory)
        tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        score1 = self.score1_h(head).unsqueeze(2).unsqueeze(3) + self.score1_t(tail).unsqueeze(1).unsqueeze(3) + self.score1_m(mid).unsqueeze(1).unsqueeze(2)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1

class TriAffineParserLinAttn(TriAffineParser):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True, reduce_last=False,
                 return_attention=False, dropout=0.0, scale="none", init_std=0.01, layer_norm=False):
        super(TriAffineParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = 150
        else:
            self._hidden_dim = attention_dim
        self.mask = mask
        
        self.score0_h = nn.Linear(self._hidden_dim, self.num_class)
        self.score0_t = nn.Linear(self._hidden_dim, self.num_class)
        self.score0_m = nn.Linear(self._hidden_dim, self.num_class)
            
        self.parser1 = TriAffine(self._hidden_dim, self.num_class, scale="none", init_std=init_std)
        
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_h1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m1 = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.uni = nn.Parameter(torch.randn(self.num_class))
        self.return_attention = return_attention
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_h = nn.LayerNorm(self._hidden_dim)
            self.norm_t = nn.LayerNorm(self._hidden_dim)
            self.norm_m = nn.LayerNorm(self._hidden_dim)
            self.norm_h1 = nn.LayerNorm(self._hidden_dim)
            self.norm_t1 = nn.LayerNorm(self._hidden_dim)
            self.norm_m1 = nn.LayerNorm(self._hidden_dim)
    
    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head = self.norm_h(head)
            tail = self.norm_t(tail)
            mid = self.norm_m(mid)
        
        score = self.score0_h(head).unsqueeze(2).unsqueeze(3) + self.score0_t(tail).unsqueeze(1).unsqueeze(3) + self.score0_m(mid).unsqueeze(1).unsqueeze(2)
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)
        
        head1 = self.linear_h1(memory)
        tail1 = self.linear_t1(memory)
        mid1 = self.linear_m1(memory)
        
        if hasattr(self, 'layer_norm') and self.layer_norm:
            head1 = self.norm_h1(head1)
            tail1 = self.norm_t1(tail1)
            mid1 = self.norm_m1(mid1)
        
        # score1 = self.scorer1(mid1).unsqueeze(1).unsqueeze(1)
        score1 = self.parser1(head1, tail1, mid1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        final_score = contract('bijkr,bijkr->bijr', alpha, score1) + uni_score
        
        if not self.return_attention:
            return final_score
        return final_score, alpha, head1, tail1, mid1

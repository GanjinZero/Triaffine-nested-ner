import torch
from torch import nn
from opt_einsum import contract
import torch.nn.functional as F


class Aligner(nn.Module):
    def __init__(self, hidden_dim):
        super(Aligner, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hs, memory, ce_mask=None):
        """
        hs: bsz * (L) * query * hidden
        memroy: bsz * seq * hidden
        ce_mask: bsz * seq
        output: bsz * (L) * query * seq
        """
        h = nn.ReLU()(self.linear(hs))
        sim = torch.bmm(h, memory.permute(0,2,1).squeeze(1)) # b * (L) * q * s
        if ce_mask is not None:
            ce_mask = ce_mask.bool()
            sim = sim.masked_fill(~ce_mask.unsqueeze(1).unsqueeze(1).expand_as(sim), value=-1e6)
        return sim
    
class Pointer(nn.Module):
    def __init__(self, hidden_dim):
        super(Pointer, self).__init__() 
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_m = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, hs, memory, ce_mask=None):
        # hs : b * (L) * q * h
        # memory: b * s * h
        # output: b * (L) * q * s
        sim = self.v(torch.tanh(self.W_h(hs).unsqueeze(-2) + \
                                self.W_m(memory).unsqueeze(1).unsqueeze(1))).squeeze(-1)
        if ce_mask is not None:
            ce_mask = ce_mask.bool()
            sim = sim.masked_fill(~ce_mask.unsqueeze(1).unsqueeze(1).expand_as(sim), value=-1e6)
        return sim

class PBiaffine(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(PBiaffine, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = 1
        self.dropout = dropout
        self.tri = nn.Parameter(torch.randn(self.num_class, self.hidden_dim, self.hidden_dim))
        self.bi0 = nn.Parameter(torch.randn(self.num_class, self.hidden_dim))
        self.bi1 = nn.Parameter(torch.randn(self.num_class, self.hidden_dim))
        self.uni = nn.Parameter(torch.randn(self.num_class))
    
    def forward(self, hs, memory, ce_mask=None):
        """
        hs: bsz * (L) * query * hidden
        memroy: bsz * seq * hidden
        ce_mask: bsz * seq
        output: bsz * (L) * query * seq
        """
        bsz, seq, hidden = memory.size()
        query = hs.size(-2)
        
        hs = hs.reshape(bsz, -1, hidden)
        
        tri_score = contract('bso,bqp,cop->bsqc', memory, hs, self.tri)
        bi_score = contract('bso,co->bsc', memory, self.bi0).unsqueeze(2) + \
                   contract('bqp,cp->bqc', hs, self.bi1).unsqueeze(1)
        uni_score = self.uni.unsqueeze(0).unsqueeze(0)
        sim = (tri_score + bi_score + uni_score).squeeze(0).reshape(bsz, seq, -1, query).permute(0,2,3,1)
        
        if ce_mask is not None:
            ce_mask = ce_mask.bool()
            sim = sim.masked_fill(~ce_mask.unsqueeze(1).unsqueeze(1).expand_as(sim), value=-1e6)
        return sim

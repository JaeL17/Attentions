import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class scaled_dot_product_attention(nn.Module):
    """
    The input consists of queries(Q) and keys(K) of dimension dk, and values(V) of dimension dv.
    1. Compute the dot products of the query with all keys, divide each by sqrt(dk)
    2. Apply a softmax function to obtain the weights on the values 
    Attention function is defined as:
        Attention(Q, K, V) = softmax(Q*T.transpose/sqrt(dk))*V
    
    Inputs: query, key, value, mask
    Outputs: context_vector, attention_distribution
    """
    def __init__(self, dim: int):
        super(scaled_dot_product_attention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor,value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attention_score = torch.bmm(query,torch.transpose(key,1,2))/self.sqrt_dim
        if mask is not None:
            attention_score.masked_fill_(mask.view(attention_score.size()), -float('Inf'))

        attention_dist = F.softmax(attention_score, -1)
        context_vec = torch.bmm(attention_dist, value)
        return context_vec, attention_dist
    
class multi_head_attention(nn.Module):
    """
    Instead of performing a single attention function with d_model-dimensional keys, values and queries,
    linearly project the queries, keys and values 'h' times with different, learned linear projections to
    d_q, d_k and d_v dimensions, respectively.
    For the projected version, perform the attention function in parallel to get d_v-dimensional output values.
    These are concatenated and projected again, resulting in the final value.
    
    MultiHead(Q, K, V) = Concat(head1, ..., head_h)W^O
    where head_i = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})
    
    # employed h = 8 parallel attention layers, or heads. For each of these we use d_k=d_v=d_model/h = 64.
    
    Inputs: query, key, value, mask
    Outputs: context_vec, attention_dist 
    """
    
    def __init__(self, d_model: int = 512, h:int = 8):
        super(multi_head_attention,self).__init__()
        
        assert d_model % h == 0, "d_model % h is not zero"
        
        self.d_head = int(d_model/ h)
        self.h = h # the number of attention heads
        self.sdpa = scaled_dot_product_attention(self.d_head)
        
        self.query_proj = nn.Linear(d_model, self.d_head * h)
        self.key_proj = nn.Linear(d_model, self.d_head * h)
        self.value_proj = nn.Linear(d_model, self.d_head * h)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] =None):
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.h, self.d_head) # B x len(Q) x N x D
        key = self.key_proj(key).view(batch_size, -1, self.h, self.d_head) # B x len(K)) x N x D
        value = self.value_proj(value).view(batch_size, -1, self.h, self.d_head) # B x len(V) x N x D
        
        # permute returns a view of the original tensor input with its dimensions permuted. 
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.h, -1, self.d_head)  # BN x len(Q) x D
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.h, -1, self.d_head)      # BN x len(K) x D
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.h, -1, self.d_head)  # BN x len(V) x D
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)
        
        context_vec, attention_dist = self.sdpa(query, key, value, mask)
        
        context_vec = context_vec.view(self.h, batch_size, -1, self.d_head)
        context_vec = context_vec.permute(1,2,0,3).contiguous().view(batch_size, -1, self.h * self.d_head)
        
        return context_vec, attention_dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

def causal_attention_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape 
        
        k = self.key(x)  
        q = self.query(x) 
        v = self.value(x) 
        
        weights = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)       
        if mask is not None:
            weights = weights.masked_fill(mask, float('-inf')) 
        attn_weights = F.softmax(weights, dim=-1)
        out = attn_weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.head_size = embed_size // num_heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            SelfAttention(embed_size, self.head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        
    def forward(self, x, mask=None):

        head_outputs = [head(x, mask) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
    


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)   
        self.register_buffer('pos_ids', torch.arange(block_size))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
            
        self.norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self,idx, targets=None): 

        B, T = idx.shape
        device= idx.device

        token_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(self.pos_ids[:T]) 
        x = token_emb + pos_emb 
        mask = causal_attention_mask(T, device)

        for block in self.blocks:
            x = block(x, mask)
            
        x = self.norm(x)  
        logits = self.lm_head(x)  
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            ).to(device)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
 
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    


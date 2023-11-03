import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attn_heads):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attn_head_size = int(hidden_size / num_attn_heads)

    def forward(self, x):
        # softmax(Qx(Kx).t)Vx
        # x: batch_size * seq_len * hidden_size
        # q: batch_size * seq_len * hidden_size
        # k: batch_size * seq_len * hidden_size
        # v: batch_size * seq_len * hidden_size
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # q: batch_size * seq_len * num_attn_heads * attn_head_size
        # k: batch_size * seq_len * num_attn_heads * attn_head_size
        # v: batch_size * seq_len * num_attn_heads * attn_head_size
        q = q.view(q.shape[0], q.shape[1], -1, self.attn_head_size)
        k = k.view(k.shape[0], k.shape[1], -1, self.attn_head_size)
        v = v.view(v.shape[0], v.shape[1], -1, self.attn_head_size)
        # q: batch_size * num_attn_heads * seq_len * attn_head_size
        # k: batch_size * num_attn_heads * attn_head_size * seq_len
        # v: batch_size * num_attn_heads * seq_len * attn_head_size
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        # attn_scores: batch_size * num_attn_heads * seq_len * seq_len
        attn_scores = torch.matmul(q, k)
        # attn_probs: batch_size * num_attn_heads * seq_len * seq_len
        attn_probs = F.softmax(attn_scores / math.sqrt(self.attn_head_size), dim=-1)
        # context: batch_size * num_attn_heads * seq_len * attn_head_size
        context = torch.matmul(attn_probs, v)
        # context: batch_size * seq_len * hidden_size 
        context = context.permute(0,2,1,3)
        context = context.contiguous().view(context.shape[0], context.shape[1], -1)

        # attn_out: batch_size * seq_len * hidden_size 
        attn_out = self.output_linear(context)
        return attn_out
        
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.intermediate(self.activation(self.dense(x)))

class Encoder(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_attn_heads=12,
                 intermediate_size=768*4,
                 ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attn_heads)
        self.ff = FeedForward(hidden_size, intermediate_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out = self.attention(x)
        ln_out = self.layer_norm(x + attn_out)
        ff_out = self.ff(ln_out)
        encoded = self.layer_norm(ff_out + ln_out)
        return encoded

if __name__ == "__main__":
    input_tensor = torch.rand([1, 512, 768])
    encoder = Encoder()
    encoder.eval()
    output = encoder(input_tensor)
    print(output.shape)

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attn_heads):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attn_head_size = int(hidden_size / num_attn_heads)

    def forward(self, x):
        # softmax(Qx(Kx).t)Vx
        # batch_size : b
        # seq_len: s
        # hidden_size : h
        # x: b * s * h
        q = self.query(x)
        k = self.query(x)
        #  reshape q : batch_size*seq_len*hidden_size -> batch_size*seq_len*attn_head_size*num_attn_heads
        q = q.reshape(q.shape[0], q.shape[1], -1, self.attn_head_size)
        q = q.transpose(q, [0, 2, 1, 3])
        k = k.reshape(k.shape[0], k.shape[1], -1, self.attn_head_size)
        k = k.transpose(k, [0, 2, 3, 1])
        e = torch.matmul(q, k)
        s = torch.softmax(e / sqrt(self.attn_head_size))
        


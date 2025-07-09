import torch
import torch.nn as nn

class BasicAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias = False)
        self.K = nn.Linear(hidden_size, hidden_size, bias = False)
        self.V = nn.Linear(hidden_size, hidden_size, bias = False)
    
    def forward(self, x):
        """
        Batch Size: b
        Sequence Length: s
        Hidden state dim: h
        Hidden state dim of 1 head: d
        Number of Heads: n

        output = softmax(XQ(XK).t())(XV)
        """
        # x : b, s, h  *  Q: h * h
        query = self.Q(x)
        # query: b, s, h

        # x : b, s, h  *  Q: h * h
        key = self.K(x)
        # key: b, s, h

        # x : b, s, h  *  V: h * h
        value = self.V(x)
        # value: b, s, h

        key_t = key.transpose(-2, -1)
        # key_t: b, h, s

        # query: b, s, h * key_t: b, h, s
        score = torch.matmul(query, key_t)
        # score = b, s, s

        prob = torch.softmax(score, dim = -1)
        # prob = b, s, s


        # prob: b, s, s * value: b, s, h
        attn = torch.matmul(prob, value)
        # attn = b, s, h

        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias = False)
        self.K = nn.Linear(hidden_size, hidden_size, bias = False)
        self.V = nn.Linear(hidden_size, hidden_size, bias = False)
        self.n_heads = num_heads
    
    def forward(self, x):
        """
        Batch Size: b
        Sequence Length: s
        Hidden state dim: h
        Hidden state dim of 1 head: d
        Number of Heads: n

        output = softmax(XQ(XK).t())(XV)
        """
        b, s, h = x.shape

        # x : b, s, h  *  Q: h * h
        query = self.Q(x)
        # query: b, s, h

        # x : b, s, h  *  Q: h * h
        key = self.K(x)
        # key: b, s, h

        # x : b, s, h  *  V: h * h
        value = self.V(x)
        # value: b, s, h

        # query, key, value: b, n_heads, s, h/n_heads
        query = query.view(b, -1, self.n_heads, h // self.n_heads).transpose(1, 2)
        key = key.view(b, -1, self.n_heads, h // self.n_heads).transpose(1, 2)
        value = value.view(b, -1, self.n_heads, h // self.n_heads).transpose(1, 2)

        key_t = key.transpose(-2, -1)
        # key_t: b, n_heads, h/n_heads, s

        score = torch.matmul(query, key_t)
        # score = b, n_head, s, s

        prob = torch.softmax(score, dim = -1)
        # prob = b, n_head, s, s

        # prob: b, n_head, s, s * value: b, n_heads, s, h/n_heads
        attn = torch.matmul(prob, value)
        # attn = b, n_head, s, h/n_heads

        attn = attn.transpose(1, 2).reshape(b, s, h)
        # attn = b, s, h

        return attn

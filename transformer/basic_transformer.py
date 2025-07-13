import torch
import torch.nn as nn
import math

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
        attention_scores = torch.matmul(query, key_t)
        # attention_scores = b, s, s

        attention_probs = torch.softmax(attention_scores, dim = -1)
        # attention_probs = b, s, s


        # attention_probs: b, s, s * value: b, s, h
        attn = torch.matmul(attention_probs, value)
        # attn = b, s, h

        return attn

class MultiHeadAttentionNaive(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
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

        attention_scores = torch.matmul(query, key_t)
        # attention_scores = b, n_head, s, s

        attention_probs = torch.softmax(attention_scores, dim = -1)
        # attention_probs = b, n_head, s, s

        # attention_probs: b, n_head, s, s * value: b, n_heads, s, h/n_heads
        attn = torch.matmul(attention_probs, value)
        # attn = b, n_head, s, h/n_heads

        attn = attn.transpose(1, 2).reshape(b, s, h)
        # attn = b, s, h

        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.WO = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.n_heads = num_heads
    
    def forward(self, x, attention_mask=None):
        """
        Batch Size: b
        Sequence Length: s
        Hidden state dim: h
        Hidden state dim of 1 head: d
        Number of Heads: n

        output = softmax(XQ(XK).t()/sqrt(hidden_size/num_heads))(XV)
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

        attention_scores = torch.matmul(query, key_t) / math.sqrt(self.hidden_size / self.n_heads)
        # attention_scores = b, n_head, s, s


        # attention_mask = b, 1, 1, s
        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = torch.softmax(attention_scores, dim = -1)
        # attention_probs = b, n_head, s, s

        # attention_probs: b, n_head, s, s * value: b, n_heads, s, h/n_heads
        attn = torch.matmul(attention_probs, value)
        # attn = b, n_head, s, h/n_heads

        attn = attn.transpose(1, 2).reshape(b, s, h)
        # attn = b, s, h

        attn = self.WO(attn)
        # attn = b, s, h

        return attn

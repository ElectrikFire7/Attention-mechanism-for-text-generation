import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, n_embd, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(n_embd, d_out_kq, bias=False)
        self.W_key = nn.Linear(n_embd, d_out_kq, bias=False)
        self.W_value = nn.Linear(n_embd, d_out_v, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        value = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/ self.d_out_kq , dim= -1)

        context_vec = attn_weights @ value
        return context_vec
    
torch.manual_seed(123)

d_in, d_out_kq, d_out_v = 3, 2, 4

sentence = 'Life is short, eat dessert first'

dc = {s:i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)

vocab_size = 50000
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

sa = SelfAttention(d_in, d_out_kq, d_out_v)
print(sa(embedded_sentence))
# Content: baby example of seq2seq
# Auther : Ryo Nakamura
# Contact: @_Ryobot on Twitter
# Date   : 2019/6/14
# This code are BSD-licensed


import torch
import torch.nn as nn

i = torch.randint(1, 10, (4, 10))
t = torch.randint(1, 10, (4, 10))
print('input:\n', i)
print('target:\n', t)

class ED(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 100)
        self.enc = nn.LSTM(100, 100, 2, batch_first=True)
        self.dec = nn.LSTM(100, 100, 2, batch_first=True)
        self.out = nn.Linear(100, 10)
        self.ls = nn.LogSoftmax(dim=-1)

    def forward(self, i, t):
        x = self.emb(i)
        _, hc = self.enc(x)
        go = torch.zeros_like(t[:, 0, None])
        t = torch.cat((go, t[:, :-1]), dim=-1)
        x = self.emb(t)
        x, _ = self.dec(x, hc)
        return self.ls(self.out(x))

    def generate(self, i):
        x = self.emb(i)
        _, hc = self.enc(x)
        o = torch.zeros_like(i[:, 0, None])
        os = []
        for _ in range(10):
            x = self.emb(o)
            x, hc = self.dec(x, hc)
            o = self.out(x).max(dim=-1)[1]
            os.append(o)
        return torch.cat(os, dim=-1)

ed = ED()
nll = nn.NLLLoss()
adam = torch.optim.Adam(ed.parameters(), lr=0.01)

for _ in range(20):
    o = ed(i, t)
    l = 0
    for j in range(10):
        l += nll(o[:, j], t[:, j])
    adam.zero_grad()
    l.backward()
    adam.step()
    print(l.item())

print('output:\n', ed.generate(i))
print('target:\n', t)

"""results
input:
tensor([[1, 9, 7, 2, 7, 3, 9, 2, 7, 9],
        [9, 6, 8, 9, 6, 1, 8, 8, 3, 7],
        [7, 6, 8, 1, 8, 7, 2, 5, 1, 1],
        [7, 9, 6, 5, 9, 4, 1, 3, 5, 5]])
target:
tensor([[2, 4, 3, 5, 7, 5, 4, 5, 5, 5],
        [6, 2, 1, 6, 7, 4, 4, 7, 8, 9],
        [7, 9, 3, 7, 6, 2, 4, 9, 6, 7],
        [3, 2, 8, 4, 6, 3, 3, 5, 9, 2]])
22.927005767822266
21.17584991455078
18.62576675415039
16.339570999145508
14.191972732543945
11.84353256225586
9.61451530456543
7.39816427230835
5.729682922363281
4.187074661254883
2.9904446601867676
2.152143955230713
1.447379231452942
0.990908145904541
0.6516780853271484
0.4599827527999878
0.3226557970046997
0.22927021980285645
0.16500318050384521
0.12299478054046631
output:
tensor([[2, 4, 3, 5, 7, 5, 4, 5, 5, 5],
        [6, 2, 1, 6, 7, 4, 4, 7, 8, 9],
        [7, 9, 3, 7, 6, 2, 4, 9, 6, 7],
        [3, 2, 8, 4, 6, 3, 3, 5, 9, 2]])
target:
tensor([[2, 4, 3, 5, 7, 5, 4, 5, 5, 5],
        [6, 2, 1, 6, 7, 4, 4, 7, 8, 9],
        [7, 9, 3, 7, 6, 2, 4, 9, 6, 7],
        [3, 2, 8, 4, 6, 3, 3, 5, 9, 2]])
"""
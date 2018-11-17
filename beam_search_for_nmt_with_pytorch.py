# Auther:  Ryo Nakamura @ Master's student at NAIST in Japan
# Contact: @_Ryobot on Twitter
#          nakamura.ryo.nm8[at]is.naist.jp
# Date:    2018/11/17
# This code are BSD-licensed


class Decoder(nn.Module):

    def __init__(self, opt):
        super().__init__(opt)
        self.max_seq_size = opt.get('max_seq_size', 32)
        vs = opt['vocab_size']
        hs = opt['hidden_size']
        nl = opt['num_layers']
        dr = opt['dropout_rate']
        
        self.embedding   = nn.Embedding(vs, hs, padding_idx=0, scale_grad_by_freq=True)
        self.rnn_decoder = nn.LSTM(hs, hs, nl, dropout=dr)
        self.projection  = nn.Linear(hs, vs, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def decode_step(self, x, hc): # x: batch, h: layer x batch x hidden
        """decode_step is 1 time step decode processing.
        """

        # hc is tuple of h and c
        x = self.embedding(x).unsqueeze(0) # x: 1 x batch x hidden
        x, hc = self.rnn_decoder(x, hc) # x: 1 x batch x hidden
        x = x.squeeze(0) # x: batch x hidden
        x = self.projection(x)
        return x, hc # x: batch x vocab, h: layer x batch x hidden

    def greedy_decode(self, hc):
        """greedy_decode is decode processing by greedy search.
        """

        bs = hc[0].size(1)
        x = self.get_go(bs) # x: batch
        vs = []
        done = [False for _ in range(bs)]
        self.step = 0
        
        while not all(done) and (self.step <= self.max_seq_size):
            self.step += 1
            x, hc = self.decode_step(x, hc) # x: batch x vocab
            x = x.max(dim=1)[1] # x: batch
            vs.append(x.tolist())
            for i in range(bs):
                if not done[i] and int(x[i]) == 3: # 3 corresponds to '<EoS>'
                    done[i] = True

        vs = list(map(list, zip(*vs))) # transpose
        return vs

    def beam_decode(self, hc):
        """beam_decode is decode processing by beam search.
        """

        bs = hc[0].size(1)
        assert bs == 1, 'Batch processing is not supported.'
        x = self.get_go(bs=1).item() # x: 1
        k = 10 # beam size excluding sentences that output '<EoS>'
        max_k = 10 # beam size
        results = []
        sequences = [([x], 0, 0)] # list of (sentence, score, process id)
        done = [False for _ in range(k)]
        self.step = 0
                
        while self.step < self.max_seq_size:
            self.step += 1
            if self.step == 2:
                nl, bs, hs = hc[0].size()
                hc = tuple([i.expand(nl, k, hs).contiguous() for i in hc])
            if 3 <= self.step:
                processes = [seq[2] for seq in sequences] # processes: topk (type: list)
                hc = tuple([i[:, processes, :] for i in hc])
            x = [seq[0][-1] for seq in sequences] # x: topk
            x = torch.LongTensor(x) # x: topk
            x, hc = self.decode_step(x, hc) # x: topk x vocab, hc: layer x topk x hidden
            p = self.log_softmax(x) # p: topk x vocab
            candidates = list()
            for i, (sent, score, proc) in enumerate(sequences):
                val, idx = p[i].topk(k=k, dim=0) # val: topk
                val = val.tolist()
                idx = idx.tolist()
                for j in range(k):
                    cand = (sent + [idx[j]], score + -val[j], i)
                    candidates.append(cand)
            ordered = sorted(candidates, key=lambda tup: tup[1])
            sequences = ordered[:k] # sequences: topk x seq (type: list)

            for seq in sequences:
                if seq[0][-1] == 3 and len(results) < max_k: # 3 corresponds to '<EoS>'
                    results.append(seq)
                    k -= 1

            for seq in results:
                if seq in sequences:
                    sequences.remove(seq)

            if k <= 0:
                break

        results = results + sequences
        results = results[:max_k]
        for seq in results:
            seq[0].pop(0) # delete '<Go>'
        vs = [results[0][0]]
        return vs

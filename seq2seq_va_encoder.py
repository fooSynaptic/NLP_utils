import collections
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
import numpy as np


PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

# 对一个序列，记录所有的词在 all_tokens 中以便之后构造词典，然后将该序列后添加 PAD 直到
# 长度变为 max_seq_len，并记录在 all_seqs 中。
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造 NDArray 实例。
def build_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
                                  reserved_tokens=[PAD, BOS, EOS])
    indicies = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indicies)


def read_data(max_seq_len):
    # in 和 out 分别是 input 和 output 的缩写。
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('./data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # 如果加上 EOS 后长于 max_seq_len，则忽略掉此样本。
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)


max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
print("Take a look at our dataset:", dataset[0])

class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是（批量大小，时间步数）。将输出互换样本维和时间步维。
        embedding = self.embedding(inputs).swapaxes(0, 1)
        emb_shape = embedding.shape
        var = nd.array(np.random.rand(emb_shape[0], emb_shape[1], emb_shape[2]))
        #print('Embedding info:', type(embedding), embedding.shape)
        #print("Embedding variation:", (embedding + var)[:2,:2, 1])
        return self.rnn(embedding + var, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
print("Output shape:", output.shape, state[0].shape)



dense = nn.Dense(2, flatten=False)
dense.initialize()
print("Initial shape:", dense(nd.zeros((3, 5, 7))).shape)

def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False,
                       flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model

def attention_forward(model, enc_states, dec_state):
    # 将解码器隐藏状态广播到跟编码器隐藏状态形状相同后进行连结。
    dec_states = nd.broadcast_axis(
        dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states)  # 形状为（时间步数，批量大小，1）。
    alpha = nd.softmax(e, axis=0)  # 在时间步维度做 softmax 运算。
    return (alpha * enc_states).sum(axis=0)  # 返回背景变量。


seq_len, batch_size, num_hiddens = 10, 4, 8
model = attention_model(10)
model.initialize()
enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
dec_state = nd.zeros((batch_size, num_hiddens))
print("Attention output method:", attention_forward(model, enc_states, dec_state).shape)


class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)

    def forward(self, cur_input, state, enc_states):
        # 使用注意力机制计算背景向量。
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # 将嵌入后的输入和背景向量在特征维连结。
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为 1。
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # 移除时间步维，输出形状为（批量大小，输出词典大小）。
        output = self.out(output).squeeze(axis=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态。
        return enc_state

def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态。
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是 BOS。
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # 我们将使用掩码变量 mask 来忽略掉标签为填充项 PAD 的损失。
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    l = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学。
        num_not_pad_tokens += mask.sum().asscalar()
        # 当遇到 EOS 时，序列后面的词将均为 PAD，相应位置的掩码设成 0。
        mask = mask * (y != out_vocab.token_to_idx[EOS])
    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            l_sum += l.asscalar()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)

#train(encoder, decoder, dataset, lr, batch_size, num_epochs)

def translate(encoder, decoder, input_seq, max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    emb = encoder.embedding(enc_input).swapaxes(0, 1)
    emb_shape = emb.shape
    var = nd.array(np.random.rand(emb_shape[0], emb_shape[1], emb_shape[2]))
    emb = emb + var

    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS:  # 当任一时间步搜索出 EOS 符号时，输出序列即完成。
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens, emb

input_seq = 'ils regardent .'
#translate(encoder, decoder, input_seq, max_seq_len)
temp = 10
res = []
embs = []
while temp:
    print("Infer step started:", temp)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs)
    r, emb = translate(encoder, decoder, input_seq, max_seq_len)
    res.append(r)
    embs.append(emb)
    temp -= 1
[print(x, '\n') for x in res]
#[print(x, emb, '\n') for x in embs]

def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        for i in range(len_pred - n + 1):
            if ' '.join(pred_tokens[i: i + n]) in ' '.join(label_tokens):
                num_matches += 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def score(input_seq, label_seq, k):
    pred_tokens, _ = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k),
                                      ' '.join(pred_tokens)))


'''
score('ils regardent .', 'they are watching .', k=2)

score('ils sont canadiens .', 'they are canadian .', k=2)
'''




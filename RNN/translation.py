import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import jieba  # 用来给中文分词
from collections import Counter
import numpy as np

torch.manual_seed(123)
device = torch.device('cpu')  # 将数据转移到cpu


# 逐句读取文本
def load_data(file_name, is_en):
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):   # enumerate 把可循环对象表示成（索引，数据）的形式
            if i > 200000:
                break
            line = line.strip()  # 去除首尾的空格和换行符
            # 在文本句子前后分别添加BOS和EOS
            if is_en:
                datas.append(['BOS'] + nltk.word_tokenize(line.lower()) + ['EOS'])
            else:
                datas.append(['BOS'] + list(jieba.cut(line, cut_all=False)) + ['EOS'])
    return datas


en_path = "news-commentary-v12.zh-en.en"
cn_path = "news-commentary-v12.zh-en.zh"
en = load_data(en_path, is_en=True)
cn = load_data(cn_path, is_en=False)


# 统计文本中每个词出现的频数，并用出现最多的词创建词典
def create_dict(sentences, max_words):
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
    most_common_words = word_count.most_common(max_words)  # 最常见的词
    word_dict = {w[0]: index + 2 for index, w in enumerate(most_common_words)}  # word2index
    word_dict["UNK"] = 0  # UNK表示词典中未出现的词
    word_dict["PAD"] = 1  # 后续句子中添加的padding
    total_words = len(most_common_words) + 2
    return word_dict, total_words


# word2index
en_dict, en_total_words = create_dict(sentences=en, max_words=50000)  # 按照出现的频次给排序{单词：频数}
cn_dict, cn_total_words = create_dict(sentences=cn, max_words=50000)
# index2word
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}


# 句子编码器：将句子的词换为字典中的index
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sorted_len):
    # 不在词典中的词UNK表示,字典的get(key,default)
    out_en_sentences = [[en_dict.get(w, en_dict["UNK"]) for w in sentence] for sentence in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, cn_dict["UNK"]) for w in sentence] for sentence in cn_sentences]
    # 按长度给英文句子排序
    if sorted_len:
        sorted_index = sorted(range(len(out_en_sentences)), key=lambda idx: len(out_en_sentences[idx]))
        # sorted_index按句子长度排序，存储的是句子对应在out_en_sentences中的索引号
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
    return out_en_sentences, out_cn_sentences


# 每个句子单词出现的频数[句子1[各单词频数],句子2[]]
en_datas, cn_datas = encode(en, cn, en_dict, cn_dict, sorted_len=True)


def get_batches(num_sentences, batch_size, shuffle=True):
    # 每个句子的在文本中的行数代替batch的索引
    batch_first_idx = np.arange(start=0, stop=num_sentences, step=batch_size)
    if shuffle:
        np.random.shuffle(batch_first_idx)
    batches = []
    for first_index in batch_first_idx:
        batch = np.arange(first_index, min(first_index + batch_size, num_sentences), 1)
        batches.append(batch)
    return batches


def add_padding(batch_sentences):
    # 给句子添加padding使其等长，并记录下句子原本的长度
    lengths = [len(sentence) for sentence in batch_sentences]
    max_len = np.max(lengths)
    data = []
    for sentence in batch_sentences:
        sen_len = len(sentence)
        sentence = sentence + [0] * (max_len - sen_len)
        data.append(sentence)

    data = np.array(data).astype('int32')
    data_lengths = np.array(lengths).astype('int32')
    return data, data_lengths


def generate_dataset(en, cn, batch_size):
    batches = get_batches(len(en), batch_size)
    datasets = []
    for batch in batches:
        batch_en = [en[idx] for idx in batch]
        batch_cn = [cn[idx] for idx in batch]
        batch_x, batch_x_len = add_padding(batch_en)
        batch_y, batch_y_len = add_padding(batch_cn)
        datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
    return datasets


batch_size = 8
datasets = generate_dataset(en_datas, cn_datas, batch_size)


# seq2seq的编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, directions, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=(directions == 2))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, batch_x, lengths):
        sorted_lengths, sorted_index = lengths.sort(0, descending=True)
        batch_x_sorted = batch_x[sorted_index.long()]
        embed = self.embedding(batch_x_sorted)
        embed = self.dropout(embed)

        # 去掉句尾的padding
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        packed_out, hidden = self.gru(packed_embed)

        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()  # contiguous 类似于深拷贝
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # 使用激活函数tanh将值范围变成（-1，1）
        return out, hidden


# Attention
class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = encoder_hidden_size
        self.dec_hidden_size = decoder_hidden_size
        self.liner_in = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size, bias=False)
        self.liner_out = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)

    def forward(self, output, context, masks):
        batch_size = output.size(0)
        y_len = output.size(1)
        x_len = context.size(1)
        x = context.view(batch_size * x_len, -1)
        x = self.liner_in(x)
        context_in = x.view(batch_size, x_len, -1)
        atten = torch.bmm(output, context_in.transpose(1, 2))  # bmm 两个Tensor类型数据相乘，类似于矩阵A*B
        atten.data.masked_fill(masks.bool(), -1e-6)
        atten = F.softmax(atten, dim=2)
        context = torch.bmm(atten, context)
        output = torch.cat((context, output), dim=2)
        output = output.view(batch_size * y_len, -1)
        output = torch.tanh(self.liner_out(output))
        output = output.view(batch_size, y_len, -1)
        return output, atten


# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.gru = nn.GRU(embed_size, dec_hidden_size, batch_first=True)

        self.liner = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_atten_masks(self, x_len, y_len):
        # 创建mask
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_masks = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]
        y_masks = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]
        mask = (~(y_masks[:, :, None] * x_masks[:, None, :])).byte()
        return mask

    def forward(self, encoder_out, x_lengths, batch_y, y_lengths, encoder_hidden):
        sorted_lengths, sorted_index = y_lengths.sort(0, descending=True)
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]
        embed = self.embedding(batch_y_sorted)
        embed = self.dropout(embed)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(),
                                                         batch_first=True)
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()
        atten_masks = self.create_atten_masks(x_lengths, y_lengths)
        out, atten = self.attention(out, encoder_out, atten_masks)
        out = self.liner(out)
        out = F.log_softmax(out, dim=-1)
        return out, hidden


# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, encoder_hid = self.encoder(x, x_lengths)
        output, hidden = self.decoder(encoder_out, x_lengths, y, y_lengths, encoder_hid)
        return output

    def translate(self, x, x_lengths, y, max_length=50):
        encoder_out, encoder_hidden = self.encoder(x, x_lengths)
        predicts = []
        batch_size = x.size(0)
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            output, hidden = self.decoder(encoder_out, x_lengths, y, y_length, encoder_hidden)
            y = output.max(2)[1].view(batch_size, 1)
            predicts.append(y)

        predicts = torch.cat(predicts, 1)
        return predicts


# 自定义损失函数，使添加的padding部分不参加计算
class MaskCriterion(nn.Module):
    def __init__(self):
        super(MaskCriterion, self).__init__()

    def forward(self, predicts, targets, masks):
        predicts = predicts.contiguous().view(-1, predicts.size(2))
        targets = targets.contiguous().view(-1, 1)
        masks = masks.contiguous().view(-1, 1)

        loss = -predicts.gather(1, targets) * masks
        loss = torch.sum(loss) / torch.sum(masks)

        return loss


dropout = 0.2
embed_size = 50
enc_hidden_size = 100
dec_hidden_size = 200
encoder = Encoder(vocab_size=en_total_words, embed_size=embed_size,
                  enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size,
                  directions=2, dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words, embed_size=embed_size,
                  enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size,
                  dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_func = MaskCriterion().to(device)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def test(mode, data):
    model.eval()
    total_words = 0
    total_loss = 0
    with torch.no_grad():
        for i, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len - 1).to(device).long()
            batch_y_len[batch_y_len <= 0] = 1

            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)

            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device)[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words
            print("Test Loss:", total_loss / total_words)


def train(model, data, epoches):
    test_datasets = []
    for epoch in range(epoches):
        model.train()
        total_words = 0
        total_loss = 0
        for it, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            if epoch != 0 and it % 10 == 0:
                test_datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
                continue
            elif it % 10 == 0:
                continue
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()

            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len - 1).to(device).long()
            batch_y_len[batch_y_len <= 0] = 1

            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)

            batch_y_len = batch_y_len.unsqueeze(1)
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device) < batch_y_len
            batch_target_masks = batch_target_masks.float()

            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            if it % 50 == 0:
                print("Epoch {} / {}, Iteration: {}, Train Loss: {}".format(epoch, epoches, it, loss.item()))
        print("Epoch {} / {}, Train Loss: {}".format(epoch, epoches, total_loss / total_words))
        if epoch != 0 and epoch % 100 == 0:
            test(model, test_datasets)


train(model, datasets, epoches=200)


# 英文翻译成中文
def en2cn_translate(sentence_id):
    en_sentence = " ".join([inv_en_dict[w] for w in en_datas[sentence_id]])
    cn_sentence = " ".join([inv_cn_dict[w] for w in cn_datas[sentence_id]])
    batch_x = torch.from_numpy(np.array(en_datas[sentence_id]).reshape(1, -1)).to(device).long()
    batch_x_len = torch.from_numpy(np.array([len(en_datas[sentence_id])])).to(device).long()
    bos = torch.Tensor([[cn_dict['BOS']]]).to(device).long()
    translation = model.translate(batch_x, batch_x_len, bos, 10)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    print(translation)
    for word in translation:
        if word != 'EOS':
            trans.append(word)
        else:
            break
    print(en_sentence)
    print(cn_sentence)
    print(" ".join(trans))


en2cn_translate(0)


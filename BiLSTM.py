import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, output_dim, n_layers, drop_prob):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)  # 确保这里定义了dropout层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM的hidden_dim要乘以2
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)  # 应用嵌入层
        lstm_out, hidden = self.lstm(embeds, hidden)  # 应用LSTM层

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # 由于是双向LSTM，我们需要合并最后一层的前向和后向hidden状态
        # 前向hidden状态
        h_forward = lstm_out[:, :self.hidden_dim]
        # 后向hidden状态
        h_backward = lstm_out[:, self.hidden_dim:]

        # 合并前向和后向的hidden状态
        lstm_out = torch.cat((h_forward, h_backward), dim=1)

        out = self.dropout(lstm_out)  # 应用dropout层
        out = self.fc(out)  # 应用全连接层
        sig_out = self.sig(out)  # 应用sigmoid激活函数

        # 取每个batch的最后一个元素作为最终输出
        sig_out = sig_out.squeeze(1)  # 压缩尺寸

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        return hidden

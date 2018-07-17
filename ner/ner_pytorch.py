import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import utils

from evaluation import precision_recall_f1

data = utils.Data()

batch_size = 32
n_epochs = 4
learning_rate = 0.005
learning_rate_decay = np.sqrt(2)
dropout_keep_probability = 0.5


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, n_tags, embedding_dim, n_hidden_rnn, pad_index):

        super(BiLSTMModel, self).__init__()

        self.hidden_dim = n_hidden_rnn
        # TODO: confirm that it's randomly intialized
        self.word_embeddings = nn.Embedding(vocab_size,
                                            embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            n_hidden_rnn)
                            # dropout=1 - dropout_keep_probability)
        self.hidden2tag = nn.Linear(n_hidden_rnn, n_tags)
        self.h, self.c = self.init_hidden()

    def init_hidden(self):
        # axis semantics : num_layers, minibatch_size, hidden_dim
        h = torch.zeros(1, 32, self.hidden_dim)
        c = torch.zeros(1, 32, self.hidden_dim)
        return h, c

    def forward(self, input_batch):
        embeds = self.word_embeddings(input_batch)
        x = embeds.view(input_batch.shape[1], input_batch.shape[0], -1)
        print('x embeds shape = {}, dtype = {}'.format(x.shape, x.dtype))
        lstm_out, (self.h, self.c) = self.lstm(x,
                                               (self.h, self.c))
        print('lstm_out shape {}'.format(lstm_out.shape))
        tag_space = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        print('tag_space shape {}, dtype = {}'.format(
            tag_space.shape, tag_space.dtype))
        log_probs = F.log_softmax(tag_space, dim=1)  # logits
        print('log probs shape {}'.format(log_probs.shape))
        return log_probs


model = BiLSTMModel(
    vocab_size=data.vocab_size,
    n_tags=21,
    embedding_dim=200,
    n_hidden_rnn=1000,
    pad_index=data.token2idx['<PAD>'],
)

loss_func = nn.NLLLoss()
# loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    for x_batch, y_batch, lengths in data.batches_generator(
            batch_size, data.train_tokens, data.train_tags):

        print('epoch')
        x_batch = torch.from_numpy(x_batch.astype('int64'))
        y_batch = torch.from_numpy(y_batch.astype('int64'))
        print('x.shape = {}, y.shape = {}'.format(x_batch.shape, y_batch.shape))

        model.zero_grad()
        model.hidden = model.init_hidden()
        print('h shape = {}, c shape = {}'.format(model.hidden[0].shape, model.hidden[1].shape))
        scores = model(x_batch)
        print('scores shape = {}'.format(scores.shape))
        print('y_batch shape = {}'.format(y_batch.shape))
        # preds = torch.argmax(scores, dim=1).reshape(y_batch.shape)
        preds = scores
        print('precs shape = {}'.format(preds.shape))
        print('dtypes preds = {}, dtypes y_batch = {}'.format(preds.dtype, y_batch.dtype))
        loss = loss_func(preds, y_batch.reshape(y_batch.shape[0] * y_batch.shape[1]))
        loss.backward(retain_graph=True)
        optimizer.step()

    learning = learning_rate / learning_rate_decay

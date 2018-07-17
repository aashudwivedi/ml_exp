import re
import numpy as np

from collections import defaultdict


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    i = 0
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            if re.match('(http|https):\/\/', token):
                tweet_tokens.append('<URL>')
            elif token.startswith('@'):
                tweet_tokens.append('<USR>')
            else:
                tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []

    i = 0
    for token in special_tokens:
        if token not in tok2idx:
            idx2tok.append(token)
            tok2idx[token] = i
            i += 1
    for token_list in tokens_or_tags:
        for token in token_list:
            if token not in tok2idx:
                idx2tok.append(token)
                tok2idx[token] = i
                i += 1
    return tok2idx, idx2tok


class Data(object):

    def __init__(self):
        self.vocab_size = 20505
        self.train_tokens, self.train_tags = read_data('data/train.txt')
        self.validation_tokens, self.validation_tags = read_data(
            'data/validation.txt')
        test_tokens, test_tags = read_data('data/test.txt')

        special_tokens = ['<UNK>', '<PAD>']
        special_tags = ['O']

        self.token2idx, self.idx2token = build_dict(
            self.train_tokens + self.validation_tokens, special_tokens)

        self.tag2idx, self.idx2tag = build_dict(self.train_tags, special_tags)

    def words2idxs(self, tokens_list):
        return [self.token2idx[word] for word in tokens_list]

    def tags2idxs(self, tags_list):
        return [self.tag2idx[tag] for tag in tags_list]

    def idxs2words(self, idxs):
        return [self.idx2token[idx] for idx in idxs]

    def idxs2tags(self, idxs):
        return [self.idx2tag[idx] for idx in idxs]

    def batches_generator(self, batch_size, tokens, tags,
                          shuffle=True, allow_smaller_last_batch=True):
        """Generates padded batches of tokens and tags."""

        n_samples = len(tokens)
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        n_batches = n_samples // batch_size
        if allow_smaller_last_batch and n_samples % batch_size:
            n_batches += 1

        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            x_list = []
            y_list = []
            max_len_token = 0
            for idx in order[batch_start: batch_end]:
                x_list.append(self.words2idxs(tokens[idx]))
                y_list.append(self.tags2idxs(tags[idx]))
                max_len_token = max(max_len_token, len(tags[idx]))

            # Fill in the data into numpy nd-arrays filled with padding indices.
            x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * \
                self.token2idx['<PAD>']
            y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * \
                self.tag2idx['O']
            lengths = np.zeros(current_batch_size, dtype=np.int32)
            for n in range(current_batch_size):
                utt_len = len(x_list[n])
                x[n, :utt_len] = x_list[n]
                lengths[n] = utt_len
                y[n, :utt_len] = y_list[n]
            yield x, y, lengths
import re

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
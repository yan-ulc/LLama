import json
from typing import Union
import numpy as np
from collections import Counter

class BPETokenizer:
    def __init__(self, min_pair_freq: int = 2, unk_token: str = "[UNK]"):
        self.vocab = {}              # Final vocab: token → id
        self.min_pair_freq = min_pair_freq
        self.unk_token = unk_token

    def init_vocab(self, corpus):
        text = " ".join([" ".join(s) for s in corpus])  # Gabung semua kalimat
        words = text.split()
        vocab = {}
        for word in words:
            chars = tuple(list(word) + ["</w>"])
            vocab[chars] = vocab.get(chars, 0) + 1
        return vocab

    def get_pair_freq(self, vocab):
        pair_freq = {}
        for token, freq in vocab.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
        return pair_freq

    def merge_pair(self, vocab, best_pair):
        new_vocab = {}
        for token, freq in vocab.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    new_token.append(token[i] + token[i+1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_vocab[tuple(new_token)] = freq
        return new_vocab

    def fit(self, corpus):
        vocab = self.init_vocab(corpus)

        while True:
            pair_freq = self.get_pair_freq(vocab)
            if not pair_freq:
                break
            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < self.min_pair_freq:
                break
            vocab = self.merge_pair(vocab, best_pair)

        # Hitung frekuensi akhir semua token hasil merge
        token_freq = Counter()
        for token_tuple, freq in vocab.items():
            for t in token_tuple:
                token_freq[t] += freq

        sorted_tokens = [token for token, _ in token_freq.most_common()]
        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.vocab[self.unk_token] = len(self.vocab)

    def encode_word(self, word):
        tokens = list(word) + ["</w>"]
        while True:
            merged = False
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]
                if pair in self.vocab:
                    tokens = tokens[:i] + [pair] + tokens[i + 2:]
                    merged = True
                    break
            if not merged:
                break

        token_ids = []
        for t in tokens:
            if t in self.vocab:
                token_ids.append(self.vocab[t])
            else:
                for c in t:
                    token_ids.append(self.vocab.get(c, self.vocab[self.unk_token]))
        return token_ids

    def encode(self, corpus):
        encoded = []
        for sentence in corpus:
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.extend(self.encode_word(word))
            encoded.append(encoded_sentence)
        return encoded

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(tid, self.unk_token) for tid in token_ids]
        words = []
        word = ""
        for token in tokens:
            if token.endswith("</w>"):
                word += token[:-4]
                words.append(word)
                word = ""
            else:
                word += token
        if word:
            words.append(word)
        return " ".join(words)

    def get_vocab(self):
        return self.vocab

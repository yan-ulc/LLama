import json 
from typing import  list ,  Union
import numpy as np

import numpy as np

class BPETokenizer:
    def __init__(self, min_pair_freq: int = 2, unk_token: str = "[UNK]"):
        self.vocab_index = {}        # dict buat nyimpan vocab index
        self.vocab = {}              # dict buat nyimpan vocab final
        self.min_pair_freq = min_pair_freq  # Frekuensi minimum pasangan agar tetap digabung
        self.unk_token = unk_token  # Token unknown jika karakter/token tidak dikenal

    def init_vocab(self, corpus):
        vocab = {}
        for sentence in corpus: 
            for word in sentence:
                chars = list(word) + ["</w>"]  # buat setiap kata dalam corpus masukkan dalam list dan tambahkan end of word 
                chars = tuple(chars) # jadikan tuple 
                vocab[chars] = vocab.get(chars, 0) + 1  # Hitung frekuensi token
        return vocab # nilai yang dikembalikan adalah vocab yang sudah dihitung frekuensinya buat setiap kata

    def get_pair_freq(self, vocab):
        pair_freq = {} # buat dict kosong untuk nampung pasangand dan frekuensinya 
        for token, freq in vocab.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])  # Ambil pasangan karakter
                pair_freq[pair] = pair_freq.get(pair, 0) + freq  # look up kedalam pair freq, dan tambahkan frekuensinya sesuai dengan frekuensi kata
        return pair_freq # nilai yang dikembalikan adalah pair freq yang sudah dihitung frekuensinya buat setiap pasangan karakter

    def merge_pair(self, vocab, best_pair):
        new_vocab = {} #buat dict kosong buat nampung hasil merge pasangan karakter
        for token, freq in vocab.items():
            new_token = []
            i = 0 
            while i < len(token):
                # Jika pasangan cocok, gabungkan dan passsing ke new token 
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    new_token.append(token[i] + token[i+1])
                    i += 2  # Lewati 2 token karena sudah digabung
                else:
                    new_token.append(token[i])
                    i += 1
            new_vocab[tuple(new_token)] = freq  # pasangan dalam token yang udah di gabung tadi passing ke new vocab
        return new_vocab

    def fit(self, corpus):
        vocab = self.init_vocab(corpus)  # ambil corpus dan masukkan ke fungsi init vocab

        while True:
            pair_freq = self.get_pair_freq(vocab)  #cek apakah 
            if not pair_freq: 
                break

            best_pair = max(pair_freq, key=pair_freq.get)  # Cari pasangan paling sering
            if pair_freq[best_pair] < self.min_pair_freq:
                break  # Hentikan jika frekuensi terlalu kecil

            vocab = self.merge_pair(vocab, best_pair)  # Gabungkan pasangan tersebut

        # Buat index untuk tuple token hasil akhir
        self.vocab_index = {token: idx for idx, token in enumerate(vocab.keys())}

        # Buat flat vocab dari semua token individual
        token_set = set()
        for token_tuple in self.vocab_index.keys():
            for t in token_tuple:
                token_set.add(t)

        self.vocab = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.vocab[self.unk_token] = len(self.vocab)  # Tambahkan token unknown

    def encode_word(self, word):
        tokens = list(word) + ["</w>"]  # Ubah kata menjadi list karakter + end token
        while True:
            merged = False
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]  # Gabungkan dua karakter berurutan
                if pair in self.vocab:
                    tokens = tokens[:i] + [pair] + tokens[i + 2:]  # Gabungkan jadi satu token
                    merged = True
                    break
            if not merged:
                break  # Jika tidak ada yang bisa digabung lagi, berhenti

        token_ids = []
        for t in tokens:
            if t in self.vocab:
                token_ids.append(self.vocab[t])  # Ambil ID dari token
            else:
                for c in t:  # Jika token tidak dikenal, encode karakter satu-satu
                    token_ids.append(self.vocab.get(c, self.vocab[self.unk_token]))
        return token_ids

    def encode(self, corpus):
        encoded = []
        for sentence in corpus:
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.extend(self.encode_word(word))  # Encode tiap kata
            encoded.append(encoded_sentence)
        return encoded

    def get_vocab(self):
        return self.vocab  # Ambil flat vocabulary akhir

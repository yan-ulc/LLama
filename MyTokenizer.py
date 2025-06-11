import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Union

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer dengan pendekatan OOP
    """
    
    def __init__(self, min_frequency: int = 2, unk_token: str = "[UNK]", end_word_token: str = "</w>"):
        """
        Inisialisasi BPE Tokenizer
        
        Args:
            min_frequency: Frekuensi minimum untuk melakukan merge
            unk_token: Token untuk kata yang tidak dikenal
            end_word_token: Token penanda akhir kata
        """
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.end_word_token = end_word_token
        
        # Vocabulary yang akan dibangun saat training
        self.vocab = {}
        self.vocab_index = {}
        self.is_trained = False
        
        # Menyimpan merge operations untuk encoding baru
        self.merge_operations = []
    
    def _get_word_tokens(self, corpus: List[List[str]]) -> Dict[Tuple, int]:
        """
        Mengubah corpus menjadi character-level tokens dengan frekuensi
        
        Args:
            corpus: List of sentences, dimana setiap sentence adalah list of words
            
        Returns:
            Dictionary dengan token tuple sebagai key dan frekuensi sebagai value
        """
        vocab = {}
        for sentence in corpus:
            for word in sentence:
                # Ubah kata menjadi karakter + end token
                char_list = list(word) + [self.end_word_token]
                char_tuple = tuple(char_list)
                
                if char_tuple in vocab:
                    vocab[char_tuple] += 1
                else:
                    vocab[char_tuple] = 1
        
        return vocab
    
    def _get_pair_frequencies(self, vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
        """
        Menghitung frekuensi pasangan karakter yang berdekatan
        
        Args:
            vocab: Dictionary token dengan frekuensinya
            
        Returns:
            Dictionary pasangan dengan frekuensinya
        """
        pair_freq = {}
        
        for token, freq in vocab.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                if pair in pair_freq:
                    pair_freq[pair] += freq
                else:
                    pair_freq[pair] = freq
        
        return pair_freq
    
    def _merge_vocab(self, vocab: Dict[Tuple, int], best_pair: Tuple[str, str]) -> Dict[Tuple, int]:
        """
        Melakukan merge pada vocabulary berdasarkan best pair
        
        Args:
            vocab: Current vocabulary
            best_pair: Pasangan yang akan di-merge
            
        Returns:
            Updated vocabulary setelah merge
        """
        new_vocab = {}
        
        for token, freq in vocab.items():
            new_token = []
            i = 0
            
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    # Merge pasangan
                    merged = token[i] + token[i+1]
                    new_token.append(merged)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            
            new_vocab[tuple(new_token)] = freq
        
        return new_vocab
    
    def _build_vocabulary_index(self, vocab: Dict[Tuple, int]) -> Dict[str, int]:
        """
        Membangun vocabulary index dari tokens yang sudah di-merge
        Tokens diurutkan berdasarkan frekuensi kemunculan (descending)
        
        Args:
            vocab: Vocabulary setelah BPE
            
        Returns:
            Dictionary mapping token ke index
        """
        # Hitung frekuensi setiap token individual
        token_freq = {}
        for token_tuple, freq in vocab.items():
            for token in token_tuple:
                if token in token_freq:
                    token_freq[token] += freq
                else:
                    token_freq[token] = freq
        
        # Sort berdasarkan frekuensi (descending), lalu alfabetis untuk tie-breaking
        sorted_tokens = sorted(token_freq.items(), key=lambda x: (-x[1], x[0]))
        
        # Buat vocabulary index berdasarkan urutan frekuensi
        vocab_dict = {}
        for idx, (token, freq) in enumerate(sorted_tokens):
            vocab_dict[token] = idx
        
        # Tambahkan UNK token di akhir
        vocab_dict[self.unk_token] = len(vocab_dict)
        
        return vocab_dict
    
    def train(self, corpus: List[List[str]]) -> None:
        """
        Training BPE tokenizer pada corpus
        
        Args:
            corpus: List of sentences untuk training
        """
        print(f"Memulai training BPE dengan min_frequency={self.min_frequency}")
        
        # Step 1: Inisialisasi vocabulary dengan character-level tokens
        vocab = self._get_word_tokens(corpus)
        print(f"Initial vocabulary size: {len(vocab)}")
        
        # Step 2: Iteratively merge most frequent pairs
        iteration = 0
        while True:
            # Hitung frekuensi pasangan
            pair_freq = self._get_pair_frequencies(vocab)
            
            if not pair_freq:
                break
            
            # Cari pasangan dengan frekuensi tertinggi
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]
            
            # Stop jika frekuensi kurang dari minimum
            if best_freq < self.min_frequency:
                break
            
            print(f"Iteration {iteration + 1}: Merging {best_pair} (freq: {best_freq})")
            
            # Simpan operasi merge
            self.merge_operations.append(best_pair)
            
            # Lakukan merge
            vocab = self._merge_vocab(vocab, best_pair)
            
            iteration += 1
        
        # Step 3: Build final vocabulary index (sorted by frequency)
        self.vocab = self._build_vocabulary_index(vocab)
        self.vocab_index = {v: k for k, v in self.vocab.items()}  # Reverse mapping
        self.is_trained = True
        
        print(f"Training selesai! Final vocabulary size: {len(self.vocab)}")
        print(f"Total merge operations: {len(self.merge_operations)}")
        
        # Print top 10 most frequent tokens
        print("\nTop 10 most frequent tokens:")
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])[:10]
        for token, idx in sorted_vocab:
            if token != self.unk_token:
                print(f"  Index {idx}: '{token}'")
    
    def encode_word(self, word: str) -> List[int]:
        """
        Encode satu kata menjadi token IDs
        
        Args:
            word: Kata yang akan di-encode
            
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer belum di-training! Jalankan train() terlebih dahulu.")
        
        # Mulai dengan character-level tokens
        tokens = list(word) + [self.end_word_token]
        
        # Apply merge operations
        for merge_pair in self.merge_operations:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_pair[0] and tokens[i+1] == merge_pair[1]:
                    # Merge
                    merged_token = tokens[i] + tokens[i+1]
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert ke IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown token dengan character fallback
                for char in token:
                    if char in self.vocab:
                        token_ids.append(self.vocab[char])
                    else:
                        token_ids.append(self.vocab[self.unk_token])
        
        return token_ids
    
    def encode_corpus(self, corpus: List[List[str]]) -> List[List[int]]:
        """
        Encode seluruh corpus menjadi token IDs
        
        Args:
            corpus: List of sentences
            
        Returns:
            List of encoded sentences
        """
        encoded = []
        for sentence in corpus:
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.extend(self.encode_word(word))
            encoded.append(encoded_sentence)
        return encoded
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs kembali menjadi text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string
        """
        if not self.is_trained:
            raise ValueError("Tokenizer belum di-training!")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab_index:
                tokens.append(self.vocab_index[token_id])
            else:
                tokens.append(self.unk_token)
        
        # Gabungkan tokens dan hapus end word markers
        text = ''.join(tokens).replace(self.end_word_token, ' ')
        return text.strip()
    
    def save_model(self, filepath: str, format: str = 'pickle') -> None:
        """
        Simpan model tokenizer
        
        Args:
            filepath: Path untuk menyimpan model
            format: Format file ('pickle' atau 'json')
        """
        if not self.is_trained:
            raise ValueError("Tokenizer belum di-training!")
        
        model_data = {
            'vocab': self.vocab,
            'vocab_index': self.vocab_index,
            'merge_operations': self.merge_operations,
            'min_frequency': self.min_frequency,
            'unk_token': self.unk_token,
            'end_word_token': self.end_word_token,
            'is_trained': self.is_trained
        }
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        elif format == 'json':
            # Convert tuple keys to strings for JSON
            json_data = model_data.copy()
            json_data['vocab_index'] = {str(k): v for k, v in self.vocab_index.items()}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("Format harus 'pickle' atau 'json'")
        
        print(f"Model disimpan ke: {filepath}")
    
    def load_model(self, filepath: str, format: str = 'pickle') -> None:
        """
        Load model tokenizer
        
        Args:
            filepath: Path file model
            format: Format file ('pickle' atau 'json')
        """
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            # Convert string keys back to int for vocab_index
            model_data['vocab_index'] = {int(k): v for k, v in model_data['vocab_index'].items()}
        else:
            raise ValueError("Format harus 'pickle' atau 'json'")
        
        self.vocab = model_data['vocab']
        self.vocab_index = model_data['vocab_index']
        self.merge_operations = model_data['merge_operations']
        self.min_frequency = model_data['min_frequency']
        self.unk_token = model_data['unk_token']
        self.end_word_token = model_data['end_word_token']
        self.is_trained = model_data['is_trained']
        
        print(f"Model berhasil di-load dari: {filepath}")
    
    def get_vocab_size(self) -> int:
        """Return ukuran vocabulary"""
        return len(self.vocab) if self.is_trained else 0
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary"""
        return self.vocab.copy() if self.is_trained else {}
    
    def get_token_frequency_stats(self) -> Dict[str, int]:
        """
        Return statistik frekuensi token untuk analisis
        
        Returns:
            Dictionary dengan token dan index-nya, diurutkan berdasarkan frekuensi
        """
        if not self.is_trained:
            raise ValueError("Tokenizer belum di-training!")
        
        token_freq = {}
        for token_id in self.vocab_index:
            token = self.vocab_index[token_id]
            token_freq[token] = 0

        for merge in self.merge_operations:
            merged_token = merge[0] + merge[1]
            if merged_token in token_freq:
                token_freq[merged_token] += 1
            else:
                token_freq[merged_token] = 1
        
        # Return vocab yang sudah terurut berdasarkan frekuensi
        return dict(sorted(self.vocab.items(), key=lambda x: x[1]))
    
    def print_vocab_stats(self, top_n: int = 20) -> None:
        """
        Print statistik vocabulary dengan frekuensi
        
        Args:
            top_n: Jumlah top tokens yang ditampilkan
        """
        if not self.is_trained:
            raise ValueError("Tokenizer belum di-training!")
        
        print(f"\nVocabulary Statistics (Top {top_n} most frequent tokens):")
        print("-" * 50)
        
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        for i, (token, idx) in enumerate(sorted_vocab[:top_n]):
            if token != self.unk_token:
                print(f"Rank {i+1:2d} | Index {idx:2d} | Token: '{token}'")
        
        if self.unk_token in self.vocab:
            unk_idx = self.vocab[self.unk_token]
            print(f"         | Index {unk_idx:2d} | Token: '{self.unk_token}' (Unknown Token)")


# Contoh penggunaan
if __name__ == "__main__":
    # Data corpus
    corpus = [
        ["aku", "suka", "dimana", "nasi"], 
        ["nasi", "dimakan", "ayam"], 
        ["makan", "dimeja", "makan"]
    ]
    
    # Inisialisasi dan training tokenizer
    tokenizer = BPETokenizer(min_frequency=2)
    tokenizer.train(corpus)
    
    # Encode corpus
    encoded_corpus = tokenizer.encode_corpus(corpus)
    print("\nEncoded corpus:")
    for i, sentence in enumerate(encoded_corpus):
        print(f"Sentence {i+1}: {sentence}")
    
    # Test encoding kata baru
    print("\nTesting encoding kata baru:")
    test_words = ["makan", "ayam", "unknown"]
    for word in test_words:
        encoded = tokenizer.encode_word(word)
        print(f"'{word}' -> {encoded}")
    
    # Print vocabulary statistics
    tokenizer.print_vocab_stats(top_n=15)
    
    # Simpan model
    tokenizer.save_model("bpe_model.pkl", format='pickle')
    tokenizer.save_model("bpe_model.json", format='json')
    
    # Test load model
    new_tokenizer = BPETokenizer()
    new_tokenizer.load_model("bpe_model.pkl", format='pickle')
    print(f"\nLoaded model vocab size: {new_tokenizer.get_vocab_size()}")
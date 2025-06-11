import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

# Tokenizer dengan BPE
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_index = {}
        
    def fit(self, corpus):
        """Train BPE tokenizer on corpus"""
        vocab = {}
        # Initialize vocabulary with characters
        for sentence in corpus:
            for word in sentence:
                chars = list(word) + ["</w>"]
                char_tuple = tuple(chars)
                vocab[char_tuple] = vocab.get(char_tuple, 0) + 1
        
        # BPE merging process
        while True:
            pair_freq = {}
            for token, freq in vocab.items():
                for i in range(len(token) - 1):
                    pair = (token[i], token[i+1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq
            
            if not pair_freq or max(pair_freq.values()) < 2:
                break
                
            best_pair = max(pair_freq, key=pair_freq.get)
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
            vocab = new_vocab
        
        # Create final vocabulary
        token_set = set()
        for token_tuple in vocab.keys():
            for token in token_tuple:
                token_set.add(token)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.vocab["[UNK]"] = len(self.vocab)
        self.vocab["[PAD]"] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        
    def encode_word(self, word):
        """Encode single word using BPE"""
        tokens = list(word) + ["</w>"]
        while True:
            merged = False
            for i in range(len(tokens)-1):
                pair = tokens[i] + tokens[i+1]
                if pair in self.vocab:
                    tokens = tokens[:i] + [pair] + tokens[i+2:]
                    merged = True
                    break
            if not merged:
                break
        
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                for char in token:
                    token_ids.append(self.vocab.get(char, self.vocab['[UNK]']))
        return token_ids
    
    def encode(self, corpus):
        """Encode entire corpus"""
        encoded = []
        for sentence in corpus:
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.extend(self.encode_word(word))
            encoded.append(encoded_sentence)
        return encoded

# RoPE (Rotary Position Embedding)
class RoPEEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            seq_len: Sequence length
        Returns:
            cos, sin: Cosine and sine embeddings
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

# Multi-Head Attention with Grouped Query Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.wk(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq_len, n_kv_heads * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat k and v for grouped query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

# SwiGLU Feed Forward Network
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        h = x + self.attention(self.attention_norm(x), mask)
        # Feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# Main LLaMA Model
class LLaMA(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        hidden_dim: int = 1376,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final normalization and output projection
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        
        # Create causal mask if not provided
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Token embeddings
        h = self.tok_embeddings(tokens)
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, mask)
        
        # Final normalization and output projection
        h = self.norm(h)
        logits = self.output(h)
        
        return logits

# Training utilities
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)

def pad_sequences(sequences, pad_token_id: int = 0):
    """Pad sequences to same length"""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded_seq = seq + [pad_token_id] * (max_len - len(seq))
        padded.append(padded_seq)
    return torch.tensor(padded, dtype=torch.long)

# Example usage and training setup
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        ["aku", "suka", "dimana", "nasi"], 
        ["nasi", "dimakan", "ayam"], 
        ["makan", "dimeja", "makan"]
    ]
    
    # Initialize and train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.fit(corpus)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Encode corpus
    encoded_corpus = tokenizer.encode(corpus)
    print(f"Encoded corpus: {encoded_corpus}")
    
    # Pad sequences
    tokens = pad_sequences(encoded_corpus, tokenizer.vocab["[PAD]"])
    print(f"Padded tokens shape: {tokens.shape}")
    
    # Initialize model
    model = LLaMA(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        max_seq_len=512
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(tokens)
        print(f"Output logits shape: {logits.shape}")
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        print(f"Output probabilities shape: {probs.shape}")
    
    # Training setup example
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Simple training loop example
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(tokens)
        
        # Prepare targets (shift by one for next token prediction)
        targets = tokens[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        
        # Calculate loss
        loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print("\nTraining completed!")
    print("\nTraining completed!")
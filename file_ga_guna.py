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

# Dataset handling for real datasets
import json
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset class for text data with proper tokenization and chunking"""
    
    def __init__(self, texts, tokenizer, max_seq_len=512, chunk_overlap=50):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunk_overlap = chunk_overlap
        
        # Process texts into chunks
        self.chunks = self._process_texts(texts)
        logger.info(f"Created {len(self.chunks)} chunks from {len(texts)} texts")
        
    def _process_texts(self, texts):
        """Process texts into overlapping chunks"""
        chunks = []
        
        for text in texts:
            # Tokenize text
            if isinstance(text, str):
                # Split by sentences or paragraphs
                sentences = text.split('. ')
                words = []
                for sentence in sentences:
                    words.extend(sentence.split())
            else:
                words = text
            
            # Encode words
            tokens = []
            for word in words:
                tokens.extend(self.tokenizer.encode_word(word))
            
            # Create overlapping chunks
            if len(tokens) <= self.max_seq_len:
                chunks.append(tokens)
            else:
                for i in range(0, len(tokens), self.max_seq_len - self.chunk_overlap):
                    chunk = tokens[i:i + self.max_seq_len]
                    if len(chunk) > self.chunk_overlap:  # Avoid too small chunks
                        chunks.append(chunk)
        
        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Pad or truncate to max_seq_len
        if len(chunk) < self.max_seq_len:
            chunk = chunk + [self.tokenizer.vocab["[PAD]"]] * (self.max_seq_len - len(chunk))
        else:
            chunk = chunk[:self.max_seq_len]
        
        return torch.tensor(chunk, dtype=torch.long)

class ModelManager:
    """Handles model saving, loading, and export"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, tokenizer, optimizer, epoch, loss, config, checkpoint_name="checkpoint"):
        """Save complete training checkpoint"""
        checkpoint_path = self.model_dir / f"{checkpoint_name}_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'tokenizer_vocab': tokenizer.vocab,
            'vocab_size': tokenizer.vocab_size
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save tokenizer separately
        tokenizer_path = self.model_dir / f"{checkpoint_name}_tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['config']
    
    def export_model(self, model, tokenizer, config, export_name="llama_model"):
        """Export model for production use"""
        export_dir = self.model_dir / f"{export_name}_export"
        export_dir.mkdir(exist_ok=True)
        
        # Save model weights
        model_path = export_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = export_dir / "tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Save configuration
        config_path = export_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save model architecture for easy loading
        model_info = {
            'vocab_size': tokenizer.vocab_size,
            'model_class': 'LLaMA',
            'config': config
        }
        
        info_path = export_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model exported to {export_dir}")
        return export_dir
    
    def load_exported_model(self, export_dir):
        """Load exported model for inference"""
        export_path = Path(export_dir)
        
        # Load model info
        with open(export_path / "model_info.json", 'r') as f:
            model_info = json.load(f)
        
        # Load tokenizer
        with open(export_path / "tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Initialize model
        config = model_info['config']
        model = LLaMA(
            vocab_size=model_info['vocab_size'],
            **config
        )
        
        # Load weights
        model.load_state_dict(torch.load(export_path / "model.pt", map_location='cpu'))
        
        return model, tokenizer, config

class DatasetLoader:
    """Handles loading various dataset formats"""
    
    @staticmethod
    def load_text_file(file_path, encoding='utf-8'):
        """Load plain text file"""
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read().split('\n')
    
    @staticmethod
    def load_json_lines(file_path):
        """Load JSONL file (one JSON per line)"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Assume text is in 'text' field, adjust as needed
                if 'text' in data:
                    texts.append(data['text'])
                elif 'content' in data:
                    texts.append(data['content'])
        return texts
    
    @staticmethod
    def load_json_file(file_path):
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return [item if isinstance(item, str) else str(item) for item in data]
        elif isinstance(data, dict):
            # Try common text fields
            for field in ['text', 'content', 'data', 'texts']:
                if field in data:
                    return data[field] if isinstance(data[field], list) else [data[field]]
        
        return [str(data)]

class Trainer:
    """Complete training pipeline"""
    
    def __init__(self, model, tokenizer, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model_manager = ModelManager()
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
        
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            
            # Prepare targets (shift by one for next token prediction)
            targets = batch[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_dataset, num_epochs, batch_size=32, save_every=5):
        """Complete training loop"""
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, epoch + 1)
            
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.model_manager.save_checkpoint(
                    self.model, self.tokenizer, self.optimizer, 
                    epoch + 1, avg_loss, self.config
                )
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model_manager.save_checkpoint(
                    self.model, self.tokenizer, self.optimizer, 
                    epoch + 1, avg_loss, self.config, "best_model"
                )
        
        # Export final model
        export_path = self.model_manager.export_model(self.model, self.tokenizer, self.config)
        return export_path

# Example usage and training setup
if __name__ == "__main__":
    # Configuration
    config = {
        'dim': 512,
        'n_layers': 8,
        'n_heads': 8,
        'n_kv_heads': 4,
        'hidden_dim': 1376,
        'max_seq_len': 512,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,
        'num_epochs': 10
    }
    
    # Example with sample data
    print("=== Testing with sample data ===")
    sample_texts = [
        "Ini adalah contoh teks dalam bahasa Indonesia untuk training model.",
        "Model LLaMA adalah arsitektur transformer yang powerful untuk language modeling.",
        "Kita akan melatih model ini dengan dataset yang lebih besar.",
        "PyTorch menyediakan tools yang bagus untuk deep learning.",
        "Tokenizer BPE membantu dalam preprocessing teks."
    ]
    
    # Initialize tokenizer
    tokenizer = BPETokenizer()
    # Convert texts to word lists for tokenizer
    corpus = [text.split() for text in sample_texts]
    tokenizer.fit(corpus)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TextDataset(sample_texts, tokenizer, max_seq_len=config['max_seq_len'])
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize model
    model = LLaMA(
        vocab_size=tokenizer.vocab_size,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        hidden_dim=config['hidden_dim'],
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, tokenizer, config)
    
    # Train model
    print("\n=== Starting training ===")
    export_path = trainer.train(dataset, num_epochs=3, batch_size=2, save_every=1)
    print(f"Model exported to: {export_path}")
    
    # Example of loading exported model
    print("\n=== Testing model loading ===")
    model_manager = ModelManager()
    loaded_model, loaded_tokenizer, loaded_config = model_manager.load_exported_model(export_path)
    print("Model loaded successfully!")
    
    # Test inference
    loaded_model.eval()
    test_text = "Ini adalah test"
    test_tokens = loaded_tokenizer.encode_word(test_text)
    test_input = torch.tensor([test_tokens], dtype=torch.long)
    
    with torch.no_grad():
        logits = loaded_model(test_input)
        probs = F.softmax(logits, dim=-1)
        print(f"Inference test completed. Output shape: {probs.shape}")
    
    print("\n=== Usage for real datasets ===")
    print("""
    # For training with real datasets:
    
    # 1. Load your dataset
    texts = DatasetLoader.load_text_file('your_dataset.txt')
    # or
    texts = DatasetLoader.load_json_lines('your_dataset.jsonl')
    
    # 2. Create larger vocabulary with your data
    corpus = [text.split() for text in texts[:10000]]  # Sample for vocab
    tokenizer = BPETokenizer()
    tokenizer.fit(corpus)
    
    # 3. Create dataset and train
    dataset = TextDataset(texts, tokenizer, max_seq_len=512)
    trainer = Trainer(model, tokenizer, config)
    export_path = trainer.train(dataset, num_epochs=50, batch_size=32)
    
    # 4. Export and use model
    # Model will be saved with all necessary files for production use
    """)
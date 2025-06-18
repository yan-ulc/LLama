import torch
import torch.nn as nn
import json
import pickle
import logging
import re
import string
from collections import Counter
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Impor & Konfigurasi
# ----------------------------------------------------------------------------
from llama_from_scratch import LLaMA as CustomLlamaBase
from llama_from_scratch import BPETokenizer as CustomBPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Konfigurasi Path ---
TOKENIZER_PKL_PATH = 'tokenizer.pkl'
MODEL_PT_PATH = 'model.pt'
CONFIG_JSON_PATH = 'config.json'
TRAIN_SQUAD_PATH = 'squad_train_qa.json'
VAL_SQUAD_PATH = 'squad_val_qa.json'
OUTPUT_MODEL_PATH = "custom_llama_final_model.pt"
OUTPUT_CHART_PATH = "evaluation_metrics.png"

# --- Konfigurasi Training ---
LEARNING_RATE = 2e-5
EPOCHS = 3
BATCH_SIZE = 8
MAX_SEQ_LEN = 256

# ... (Class TokenizerWrapper tetap sama) ...
class TokenizerWrapper:
    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f: self.tokenizer = pickle.load(f)
        if not hasattr(self.tokenizer, 'id_to_token'): self.tokenizer.id_to_token = {idx: token for token, idx in self.tokenizer.vocab.items()}
        self._add_special_tokens()
    def _add_special_tokens(self):
        special_tokens = ['[BOS]', '[EOS]']; vocab = self.tokenizer.vocab
        for token in special_tokens:
            if token not in vocab: vocab[token] = self.tokenizer.vocab_size; self.tokenizer.id_to_token[self.tokenizer.vocab_size] = token; self.tokenizer.vocab_size += 1
        self.bos_id = vocab['[BOS]']; self.eos_id = vocab['[EOS]']; self.pad_id = vocab['[PAD]']
    def encode(self, text: str, add_special_tokens=True):
        words = text.split(); token_ids = [item for word in words for item in self.tokenizer.encode_word(word)]
        if add_special_tokens: return [self.bos_id] + token_ids + [self.eos_id]
        return token_ids
    def decode(self, token_ids):
        ids_to_decode = [i for i in token_ids if i not in [self.bos_id, self.eos_id, self.pad_id]]
        tokens = [self.tokenizer.id_to_token.get(tid, "[UNK]") for tid in ids_to_decode]
        return ''.join(tokens).replace("</w>", " ").strip()

# ... (Fungsi preprocess_generative tetap sama) ...
def preprocess_generative(examples, tokenizer):
    IGNORE_INDEX = -100; all_input_ids, all_labels, all_attention_masks = [], [], []
    prompts = examples['input']; targets = examples['target']
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False); target_ids = tokenizer.encode(target, add_special_tokens=False)
        input_ids = [tokenizer.bos_id] + prompt_ids + target_ids + [tokenizer.eos_id]
        labels = [IGNORE_INDEX] * (len(prompt_ids) + 1) + target_ids + [tokenizer.eos_id]
        all_input_ids.append(torch.tensor(input_ids)); all_labels.append(torch.tensor(labels))
    max_len = MAX_SEQ_LEN
    padded_inputs = torch.full((len(all_input_ids), max_len), tokenizer.pad_id, dtype=torch.long)
    padded_labels = torch.full((len(all_input_ids), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_masks = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(all_input_ids):
        len_ids = min(len(ids), max_len)
        padded_inputs[i, :len_ids] = ids[:len_ids]
        padded_labels[i, :len_ids] = all_labels[i][:len_ids]
        attention_masks[i, :len_ids] = 1
    return {'input_ids': padded_inputs, 'attention_mask': attention_masks, 'labels': padded_labels}


# ----------------------------------------------------------------------------
# BAGIAN BARU: FUNGSI UNTUK MENGHITUNG METRIK EM & F1
# ----------------------------------------------------------------------------
def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def generate_text_for_eval(model, tokenizer, prompt_text, device):
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    # Potong prompt jika terlalu panjang untuk dimasukkan ke model
    prompt_ids = prompt_ids[:MAX_SEQ_LEN]
    input_tensor = torch.tensor([prompt_ids]).to(device)

    with torch.no_grad():
        for _ in range(30): # Batasi panjang jawaban maksimal
            logits = model(tokens=input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    return tokenizer.decode(input_tensor[0].tolist())


# ----------------------------------------------------------------------------
# FUNGSI EVALUASI YANG DIPERBARUI
# ----------------------------------------------------------------------------
def evaluate(model, dataloader, loss_function, tokenizer, device):
    logger.info("Memulai evaluasi...")
    model.eval()
    total_loss = 0
    total_f1 = 0
    total_em = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Hitung Loss
            logits = model(tokens=input_ids, mask=None)
            loss = loss_function(logits.view(-1, model.vocab_size), labels.view(-1))
            total_loss += loss.item()

            # Hitung EM & F1 dengan membandingkan teks hasil generate
            for i in range(len(batch['input'])):
                prompt = batch['input'][i]
                reference = batch['target'][i]

                generated_text = generate_text_for_eval(model, tokenizer, prompt, device)

                total_em += compute_exact_match(generated_text, reference)
                total_f1 += compute_f1(generated_text, reference)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_em = total_em / num_samples
    avg_f1 = total_f1 / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    model.train()
    return avg_loss, perplexity.item(), avg_em, avg_f1

# ----------------------------------------------------------------------------
# Fungsi Utama
# ----------------------------------------------------------------------------
# GANTI SELURUH FUNGSI main() ANDA DENGAN YANG INI

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Menggunakan device: {device}")

    # --- Muat Config & Tokenizer ---
    with open(CONFIG_JSON_PATH, 'r') as f:
        config_dict = json.load(f)
    config_dict['max_seq_len'] = MAX_SEQ_LEN

    tokenizer = TokenizerWrapper(pkl_path=TOKENIZER_PKL_PATH)
    config_dict['vocab_size'] = tokenizer.tokenizer.vocab_size

    # --- Muat Model ---
    model = CustomLlamaBase(**config_dict)

    # --- PERBAIKAN: BLOK KODE UNTUK MENGATASI SIZE MISMATCH DIMASUKKAN KEMBALI ---
    logger.info("Memuat bobot pre-trained dari model.pt secara parsial...")
    pretrained_dict = torch.load(MODEL_PT_PATH, map_location='cpu')

    # Hapus bobot dari layer yang ukurannya tidak cocok
    pretrained_dict.pop('tok_embeddings.weight', None)
    pretrained_dict.pop('output.weight', None)

    # Muat sisa bobot yang cocok. `strict=False` mengizinkan ini.
    model.load_state_dict(pretrained_dict, strict=False)
    logger.info("Bobot untuk layer Transformer berhasil dimuat. Embedding & output layer akan dilatih dari awal.")
    # --- AKHIR PERBAIKAN ---

    model.to(device)

    # --- Muat Dataset (termasuk validasi) ---
    def load_json_list(file_path):
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)

    train_data_list = load_json_list(TRAIN_SQUAD_PATH)
    val_data_list = load_json_list(VAL_SQUAD_PATH)

    # Gunakan seluruh data training sekarang
    train_dataset = Dataset.from_list(train_data_list)
    val_dataset = Dataset.from_list(val_data_list)
    # Preprocess kedua dataset
    train_processed = train_dataset.map(
        lambda x: preprocess_generative(x, tokenizer),
        batched=True,
        remove_columns=['input', 'target']  # Hapus kolom dari data training (INI BENAR)
    )
    val_processed = val_dataset.map(
        lambda x: preprocess_generative(x, tokenizer),
        batched=True  # JANGAN HAPUS kolom dari data validasi (INI PERBAIKANNYA)
    )

    # Set format setelahnya
    train_processed.set_format(type='torch')
    val_processed.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'input', 'target'])

    train_dataloader = DataLoader(train_processed, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_processed, batch_size=BATCH_SIZE)

    # --- Setup Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Training & Evaluation Loop ---
    history = {'train_loss': [], 'val_loss': [], 'perplexity': [], 'em': [], 'f1': []}
    logger.info("=== MEMULAI TRAINING & EVALUASI ===")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(tokens=input_ids, mask=None)
            loss = loss_function(logits.view(-1, model.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        logger.info(f"Loss Training Rata-rata Epoch {epoch + 1}: {avg_train_loss:.4f}")

        # Lakukan evaluasi setelah setiap epoch
        val_loss, perplexity, em, f1 = evaluate(model, val_dataloader, loss_function, tokenizer, device)
        history['val_loss'].append(val_loss)
        history['perplexity'].append(perplexity)
        history['em'].append(em)
        history['f1'].append(f1)
        logger.info(f"Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f} | Exact Match: {em:.4f} | F1 Score: {f1:.4f}")

    # --- Plotting Grafik ---
    logger.info("Training selesai. Membuat grafik metrik evaluasi...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss / Perplexity', color=color)
    ax1.plot(history['train_loss'], 'r--', label='Training Loss')
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
    ax1.plot(history['perplexity'], 'r:', label='Perplexity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('F1 / Exact Match', color=color)
    ax2.plot(history['f1'], 'b--', label='F1 Score')
    ax2.plot(history['em'], 'b-', label='Exact Match')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Metrik Training & Evaluasi per Epoch')
    plt.savefig(OUTPUT_CHART_PATH)
    logger.info(f"Grafik berhasil disimpan di: {OUTPUT_CHART_PATH}")

    # --- Simpan Model Final ---
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    logger.info(f"Model berhasil disimpan di: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
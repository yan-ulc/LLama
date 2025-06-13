import torch
import pickle
import json
from llama_from_scratch import LLaMA
from llama_from_scratch import BPETokenizer


# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
if not hasattr(tokenizer, "id_to_token"):
    tokenizer.id_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
# Load config
with open("config.json", "r") as f:
    config = json.load(f)
    print(config)


# Load model kerangka + bobot
model = LLaMA(
    vocab_size=config["vocab_size"],
    dim=config["dim"],
    n_layers=config["n_layers"],
    n_heads=config["n_heads"],
    n_kv_heads=config["n_kv_heads"],
    hidden_dim=config["hidden_dim"], 
    max_seq_len=config["max_seq_len"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"]
)

model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

# Fungsi prediksi kata selanjutnya
def predict_next_word(text, top_k=5):
    input_ids = tokenizer.encode(text)  # Ubah teks jadi list of ID

    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]
# Kalau input_ids = [1, 57]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)  # Output shape: (1, seq_len, vocab_size)
    print("input_ids:", input_ids)
    print("shape:", torch.tensor(input_ids).shape)


    # Ambil logits dari token terakhir
    last_logits = logits[0, -1]  # Shape: (vocab_size)
    probs = torch.softmax(last_logits, dim=-1)

    # Ambil top-k prediksi
    top_probs, top_indices = torch.topk(probs, k=top_k)

    print(f"\nInput: '{text}'")
    print("Top next word predictions:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        word = tokenizer.decode([token_id])  # Ubah ID ke teks
        print(f"{i+1}. {word} (prob: {top_probs[i].item():.4f})")

# Run
if __name__ == "__main__":
    user_input = input("Masukkan teks awal: ")
    predict_next_word(user_input)

import numpy as np
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)    

def SwingGlu(x1,x2):
    return  x2 * swish(x1) 

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # Untuk stabilitas numerik
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

  
def rope(x):
    batch_size, seq_len, d = x.shape
    setengah_dim = dim // 2

    result = np.zeros_like(x)
    for i in range(0,d,2):
        x0 = x[i]
        x1 = x[i+1] 
        theta = position / (10000 ** (i / d))
        cos = np.cos(theta)
        sin = np.sin(theta)
        result[i] = x0 * cos - x[i] * sin
        result[i+1] = x0 * sin + x[i] * cos
    return result

# def RMSnorm(x, gamma = 0.5):
#     output = []
#     for kalimat in range(len(x)):
#         kalimat_norm =[]
#         for kata in range(len(x[kalimat])):
#             sum_sqr = 0   
#             kata = x[kalimat][kata]
#             vektor_kata =sum( v**2 for v in kata)
#             sum_sqr += vektor_kata
#             rms = sqrt(sum_sqr/len(kata))
#             norm = [gamma * v / rms for v in kata]
#             kalimat_norm.append(norm)
#         output.append(kalimat_norm)
#     return output
def split_head(x):
        reshape = x.reshape(x.shape[0], num_head, -1)
        return  reshape.transpose(1, 0, 2)          
def RMSnorm(x, gamma = 0.5):
   x = np.array(x)
   rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
   norm = gamma * x / rms
   return norm

# def MHSE(x, wq, wk, wv, wo, num_head, kv_cache):
#     Q = matmul(x, wq.T)
#     K = matmul(x, wk.T)
#     V = matmul(x, wv.T)
# # Tambah Positional Encoding
#     for i in range(len(Q)):
#         Q[i] += rope(Q[i], i)
#         K[i] += rope(K[i], i)
# # Splt Ke multihead
#     Q = split_head(Q)
#     K = split_head(K)
#     V = split_head(V)
# # Update kv cache
#     K_cache, V_cache = kv_cache
#     K = np.concatenate([K_cache, K], axis=1)
#     V = np.concatenate([V_cache, V], axis=1)
#     #hitung attention
#     score = matmul(Q, np.transpose(K,(0,2,1))) / np.sqrt(d)
#     weight = softmax(score, axis=-1)
#     context = matmul(weight, V)
#     context = context.transpose(1, 0, 2).reshape(x.shape[0], -1)
#     output = matmul(context, wo.T)
#     return output, (K, V)

def GMHA(x, wq, wk_group, wv_group, wo, num_head, num_group):
    dim = x.shape[1]
    dim_per_group= dim // num_group
    seq_len = x.shape[0]
    heads_per_group= num_head // num_group
    head_dim = dim // num_head

    #Proyeksi Querry untuk semua head
    Q = matmul(x, wq.T)
    Q = Q.reshape(seq_len, num_head, head_dim).transpose(1, 0, 2)
    # siapkan list untuk key dan value yang akan dibagikan ke tiap group
    K_all = []
    V_all = []
    #Proses tiap grup
    for g in range(num_group):
        x_group = x[:, g * dim_per_group : (g+1) * dim_per_group]
        K_group = matmul(x_group, wk_group[g].T)
        K_group = K_group.reshape(heads_per_group, seq_len,head_dim)
        V_group = matmul(x_group, wv_group[g].T)
        V_group = V_group.reshape(heads_per_group, seq_len,head_dim)
        
        K_all.append(K_group)
        V_all.append(V_group)

    # Gabungkan hasil dari semua grup    
    K = np.concatenate(K_all, axis=0)
    V = np.concatenate(V_all, axis=0)
    #hitung attention
    score = matmul(Q, np.transpose(K,(0,2,1))) / np.sqrt(head_dim)
    weight = softmax(score, axis=-1)
    context = matmul(weight, V)
    context = context.reshape(seq_len, num_head*head_dim)
    output = matmul(context, wo.T)
    return output
    
def Feedforward(x):
    matmul = np.matmul
    x = np.array(x)
    dx = x.shape[1]
    dx = dx *4
    w1 = np.random.randn(len(x[0]),dx)
    b1 = np.random.randn(len(x[0]),dx)
    Logits = matmul(x, w1) + b1
    belah = Logits.shape[1]//2
    x1 = Logits[:,:belah]
    x2 = Logits[:,belah:]
    Logits = SwingGlu(x1,x2)
    w2 = np.random.randn(*Logits.shape)
    b2 = np.random.randn()
    linear = matmul(Logits, w2.T) + b2
    output = linear
    return output



def tokenizer(corpus):
    vocab = {}
    for kalimat in corpus:
        for kata in kalimat:
            char = list(kata) + ["</w>"]
            char = tuple(char)
            if char in vocab:
                vocab[char] += 1
            else:
                vocab[char] = 1
    while True:
        pair_freq = {}
        for token, freq in vocab.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                if pair in pair_freq:
                    pair_freq[pair] += freq
                else:
                    pair_freq[pair] = freq
        best_value = max(pair_freq.values())
        best_pair = max(pair_freq, key=pair_freq.get)

        if best_value < 2:
            break

        new_vocab = {}
        for token, freq in vocab.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    new_token.append(token[i] + token[i+1])
                    new_vocab[tuple(new_token)] = freq
                    i += 2

                else:
                    new_token.append(token[i])
                    i += 1
                new_vocab[tuple(new_token)] = freq

        vocab = new_vocab
        vocab_index = {}
        for i, token in enumerate(vocab.keys()):
            vocab_index[token] = i
        
    return vocab_index

def urut_vocab(vocab_index):
    token_set = set()
    for token_tuple in vocab_index.keys():
        for t in token_tuple:
            token_set.add(t)
            
    vocab_dict = {token: idx for idx, token in enumerate(sorted(token_set))}
    vocab_dict["[UNK]"] = len(vocab_dict)  # optional: token unknown
    return vocab_dict



def encode_word(word, vocab):
    tokens = list(word) + ["</w>"]
    while True:
        merged = False
        for i in range(len(tokens)-1):
            pair = tokens[i] + tokens[i+1]
            if pair in vocab:
                tokens = tokens[:i] + [pair] + tokens[i+2:]
                merged = True
                break
        if not merged:
            break

    token_ids = []
    for t in tokens:
        if t in vocab:
            token_ids.append(vocab[t])
        else:
            for c in t:
                if c in vocab:
                    token_ids.append(vocab[c])
                else:
                    token_ids.append(vocab['[UNK]'])
    return token_ids

def encode_corpus(corpus, vocab):
    encoded = []
    for sentence in corpus: 
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.extend(encode_word(word, vocab))
        encoded.append(encoded_sentence)
    return encoded

def padding(token,max_len, padding_value = 0):
    token = list(token)
    padded =np.full((len(token),max_len), padding_value)

    for i , token in enumerate(token): 
        p_token = len(token)
        padded[i, :p_token] = token
    return padded 

def buat_batch(token, batch_size = 10):
    return [token[i:i+batch_size]  for i in range(0,len(token),batch_size)]
dim = 512
d = 64 #dimensi per head8
num_head = 8
matmul = np.matmul
data = pd.read_csv("file_baru.csv")
data["Clean"] = data["Text Tweet"].apply(clean_text) 
texts = data["Clean"].tolist()
labels =  data["Sentiment"].tolist()
corpus = []
for text in texts:
    words = text.split()    x
    corpus.append(words)

vocab_index = tokenizer(corpus)
vocab  = urut_vocab(vocab_index)
token  = encode_corpus(corpus,vocab)
max_len = max(len(i) for i in token)
token = padding(token,max_len)
batch = buat_batch(token)
batch_size = len(batch[0])

print("ayam")















# gmha = GMHA(norm, wq, wk_group, wv_group, wo, num_head, num_group)
# print(gmha.shape)










    






import numpy as np
np.random.seed()
corpus = [["aku", "suka", "dimana", "nasi"], ["nasi", "dimakan", "ayam"], ["makan", "dimeja" ,"makan"]]
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

    
vocab_index = tokenizer(corpus)
print(vocab_index)
    
def urut_vocab(vocab_index):
    token_set = set()
    for token_tuple in vocab_index.keys():
        for t in token_tuple:
            token_set.add(t)
            
    vocab_dict = {token: idx for idx, token in enumerate(sorted(token_set))}
    vocab_dict["[UNK]"] = len(vocab_dict)  # optional: token unknown
    return vocab_dict

vocab = urut_vocab(vocab_index)
print(len(vocab))
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

token = encode_corpus(corpus, vocab)
print(token)

import numpy as np
np.random.seed()

def padding(token,max_len, padding_value = 0):
    token = list(token)
    padded =np.full((len(token),max_len), padding_value)

    for i , token in enumerate(token): 
        p_token = len(token)
        padded[i, :p_token] = token
    return padded 
max_len = max(len(i) for i in token)
token = padding(token,max_len)
print(token.shape)

d = 16
e_matriks = np.array(np.random.rand(len(vocab), d))
embedding = e_matriks[token]
print(embedding.shape)

x = np.array(embedding)
eps = 1e-6
rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps )
norm = x/rms
hasil = norm * 0.5
print(hasil.shape)

def rope(Q,K):
    assert Q.shape == K.shape
    batch, num_group, num_head, seq_len, head_dim= Q.shape
    assert head_dim % 2== 0
    set_dim = head_dim//2
    freq   = 1/ 1000 **(np.arange(0,set_dim)/set_dim)
    pos = np.arange(seq_len)
    angle = np.outer(pos,freq)
    sin = np.sin(angle)[None,None,None,:,:]
    cos = np.cos(angle)[None,None,None,:,:]

    def rotate(x):
        X1 = x[...,::2]
        X2 = x[...,1::2]
        rotasi = np.concatenate([X1*cos - X2*sin, X1*sin + X2*cos], axis=-1)
        return rotasi

    Q = rotate(Q)
    
    K = rotate(K)
    return (Q,K)    


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # Untuk stabilitas numerik
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
num_group = 2
num_head = 2
head_dim = 4 


wq = np.random.randn(d,d)
wk = np.random.randn(d,d)
wv = np.random.randn(d,d)

Q = hasil @ wq
K = hasil @ wk
V = hasil @ wv

 
Q = Q.reshape(3,8,num_group, num_head, head_dim)
K = K.reshape(3,8,num_group, num_head, head_dim)
V = V.reshape(3,8,num_group, num_head, head_dim)

Q = Q.transpose(0,2,3,1,4)
K = K.transpose(0,2,3,1,4)
V = V.transpose(0,2,3,1,4)


Q,K = rope(Q, K)



scores = np.einsum('bghqd, bghkd->bghqk', Q,K)
print (scores.shape)
b,g,h,s,s = scores.shape
mask = np.tril(np.ones((s,s), dtype=bool))
mask = mask[None, None, None, :,:]
scores = np.where(mask, scores, -1e9)

scores = scores/ np.sqrt(head_dim)
weight =  softmax(scores, axis=-1)
output = np.einsum("bghqk, bghkd -> bghqd", weight, V)
print (weight.shape)
output = output.transpose(0,3,1,2,4)
print(output.shape)
b, s, g,h,d = output.shape
output = output.reshape(b,s,g*h*d)
print(output.shape)

def RMSnorm(x, gamma = 0.5):
   x = np.array(x)
   rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
   norm = gamma * x / rms
   return norm

add1 =output +  embedding
norm = RMSnorm(add1)
print(norm.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)    

def SwingGlu(x1,x2):
    return  x2 * swish(x1) 
matmul = np.matmul
x = np.array(norm)
dx = x.shape[1]
dx = dx *4
b, s,d = norm.shape
w1 = np.random.randn( d, s, 4*d)


# Logits = matmul(x, w1) + b1
Logits = np.einsum("hij,jik->hik", x,w1)
b1 = np.random.randn(*Logits.shape)
Logits = Logits + b1
print(Logits.shape ,"before")
belah = Logits.shape[-1]//2
x1 = Logits[:,:, :belah]
x2 = Logits[:,:,belah:]
logits = SwingGlu(x1,x2)
print(logits.shape,"after")
w2 = np.random.randn(logits.shape[2],s,logits.shape[2]//2 )
linear_out = np.einsum("hij,jik->hik",logits, w2)
b2 = np.random.randn(*linear_out.shape)
linear_out = linear_out + b2
print(linear_out.shape,"linear")

add2 = linear_out + add1
nnorm = RMSnorm(add2)
print(nnorm.shape)

Linear_w= np.random.randn(d, len(vocab))
b = np.random.randn(len(vocab))
last_linear = np.einsum( "abc,ce->abe", nnorm, Linear_w)
print(last_linear.shape)

model = softmax(last_linear)
print(model.shape)

sums= np.sum(model, axis =-1)
print(sums.shape)




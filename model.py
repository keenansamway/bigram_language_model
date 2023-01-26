import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iterations = 3000
eval_interval = 300
learning_rate = 1e-3
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iterations = 200
n_embed = 32

torch.manual_seed(42)

with open('datasets/harry_potter_books/cleaned_book.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda e : [stoi[ch] for ch in e]
decode = lambda d : ''.join([itos[i] for i in d])

data = torch.tensor(encode(text), dtype=torch.long)

train_val_split = int(0.9*len(data))
train_data = data[:train_val_split]
val_data = data[train_val_split:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) / (C**-0.5)
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_encoding = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embed = self.embedding_table(idx) #(B,T,n_embed)
        pos_encod = self.positional_encoding(torch.arange(T, device=device)) #(T,n_embed)
        x = token_embed + pos_encod #(B,T,n_embed)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_length):
        for _ in range(max_length):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    
model = BigramLM(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter} train loss {losses['train']:.3f} val loss {losses['val']:.3f}")
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
out = model.generate(context,max_length=500)[0].tolist()
print(decode(out))
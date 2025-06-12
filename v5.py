import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #how many independent sequences will we process in parallel?
block_size = 256 #what is the maximum context length for predictions?
max_iters = 5000 #how many iterations in total to train for?
eval_interval = 500 #how often to evaluate the loss on the validation set?
learning_rate = 3e-4 #the initial learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #how many iterations to evaluate the loss on the validation set?
n_embd = 384 #dimension of the token embeddings
n_head =6
n_layer =6
dropout=0.2 #dropout rate to prevent overfitting
# head_size=16 #dimension of each attention head

# -----------

torch.manual_seed(1337) #for reproducibility

# tiny shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers and back
stoi = { ch:i for i,ch in enumerate(chars) } # string to integer
itos = { i:ch for i,ch in enumerate(chars) } # integer to string
# encode the text into integers
encode = lambda s: [stoi[c] for c in s] # encode a string into a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers into a string

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix for causal attention
        
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting  
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C) -> (B,T,head_size)
        q = self.query(x) # (B,T,C) -> (B,T,head_size)
    
        # compute attention scores("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # apply causal mask
        wei = F.softmax(wei, dim=-1) # normalize to probabilities
        wei = self.dropout(wei) # apply dropout to the attention weights
        #perform the weighted aggregation of the values
        v=self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # create n_heads instances of Head
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

    def forward(self, x):
        out= torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of all heads
        out = self.dropout(self.proj(out)) # project back to the original dimension
        return out # (B,T,C) -> (B,T,n_embd)
        
        
class FeedForward(nn.Module):
    """a simple MLP followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # first linear layer expands the dimension and 4 hidden units
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # second linear layer projects back to the original dimension
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """transformer block: commnication followed by computation
    This block consists of a multi-head self-attention layer followed by a feedforward layer."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer pre-normalization after self-attention
        self.ln2 = nn.LayerNorm(n_embd) # layer pre-normalization after feedforward layer
        
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # residual connection around self-attention
        x = x + self.ffwd(self.ln2(x)) # residual connection around feedforward layer
        return x

#super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd) # positional embeddings for each position in the block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # stack 4 transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to project to vocab size
        
    def forward(self, idx, targets=None):
        B,T = idx.shape # B is batch size, T is block size
        
        # idx and targets are both (batch_size, block_size) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding(torch.arange(T, device=idx.device)) # (T,C)
        
        x = tok_emb + pos_emb # add positional embeddings to token embeddings

        x = self.blocks(x) # pass through the transformer blocks
        x = self.ln_f(x) # apply final layer normalization
        logits = self.lm_head(x) # (B,T,C) -> (B,T,vocab_size)        
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # reshape to (B*T, C)
            targets = targets.view(B * T) # reshape to (B*T,)
            loss = F.cross_entropy(logits, targets) # compute the loss
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

        
model = BigramLanguageModel()
m = model.to(device)

#no of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)

    # backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate some samples
context = torch.zeros((1, 1), dtype=torch.long, device=device) # start with a single token
print("Generating text...")
generated = m.generate(context, max_new_tokens=3000)[0].tolist()
generated_text = decode(generated)
print(generated_text)

# save the output to a text file
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
print("Generated text saved to output.txt")
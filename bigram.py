import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #how many independent sequences will we process in parallel?
block_size = 8 #what is the maximum context length for predictions?
max_iters = 5000 #how many iterations in total to train for?
eval_interval = 100 #how often to evaluate the loss on the validation set?
learning_rate = 1e-3 #the initial learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #how many iterations to evaluate the loss on the validation set?

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

#super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (batch_size, block_size, vocab_size)
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            
model = BigramLanguageModel(vocab_size)
m = model.to(device)

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
generated = m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
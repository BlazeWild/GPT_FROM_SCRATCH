# GPT From Scratch 🚀

A minimal implementation of a GPT (Generative Pre-trained Transformer) model built from scratch using PyTorch. This project demonstrates the core concepts of transformer-based language models by training a character-level GPT on the Tiny Shakespeare dataset.

## 📋 Overview

This repository contains a step-by-step implementation of a GPT model, starting from a simple bigram model and progressively building up to a full transformer architecture with multi-head self-attention, feed-forward networks, and layer normalization.

## 🎯 Features

- **Character-level tokenization** - Simple character-based vocabulary
- **Transformer architecture** - Multi-head self-attention and feed-forward layers
- **Positional embeddings** - Learnable position encodings
- **Layer normalization** - For training stability
- **Dropout regularization** - To prevent overfitting
- **Training on Tiny Shakespeare** - ~1MB of Shakespeare text data

## 📁 Project Structure

```
├── bigram.py          # Simple bigram baseline model
├── v2.py             # Basic transformer with single attention head
├── v3.py             # Multi-head attention implementation
├── v4.py             # Added feed-forward networks
├── v5.py             # Complete GPT with layer norm and dropout
├── input.txt         # Tiny Shakespeare dataset
├── output.txt        # Generated text samples
├── trainiglogs.md    # Training logs and performance metrics
└── test/
    ├── input.txt     # Test dataset
    └── test.ipynb    # Jupyter notebook for experiments
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/BlazeWild/GPT_FROM_SCRATCH.git
cd GPT_FROM_SCRATCH
```

2. Install dependencies:

```bash
pip install torch
```

### Training

Run the complete GPT model:

```bash
python v5.py
```

Or start with the simple bigram model:

```bash
python bigram.py
```

## 🏗️ Model Architecture

The final model (`v5.py`) implements a complete GPT architecture with:

- **Embedding dimension**: 384
- **Number of attention heads**: 6
- **Number of transformer layers**: 6
- **Context length**: 256 characters
- **Vocabulary size**: 65 unique characters
- **Parameters**: ~10.8M

### Hyperparameters

```python
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

## 📊 Performance

Training on 1x L40S GPU:

- **Training time**: ~8 minutes
- **Final train loss**: 1.11
- **Final validation loss**: 1.47
- **Parameters**: 10.788929M

### Training Progress

```
Step 0: train loss 4.2848, val loss 4.2827
Step 1000: train loss 1.5967, val loss 1.7782
Step 2000: train loss 1.3411, val loss 1.5675
Step 3000: train loss 1.2267, val loss 1.5038
Step 4000: train loss 1.1473, val loss 1.4858
Step 4500: train loss 1.1105, val loss 1.4734
```

## 🎭 Sample Generation

After training, the model can generate Shakespeare-like text:

```python
# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
```

## 🧩 Implementation Details

### 1. Bigram Model (`bigram.py`)

- Simplest baseline using only token embeddings
- Each token predicts the next token independently

### 2. Self-Attention (`v2.py`)

- Introduces single-head self-attention
- Tokens can now "communicate" with each other

### 3. Multi-Head Attention (`v3.py`)

- Multiple attention heads running in parallel
- Allows the model to attend to different aspects

### 4. Feed-Forward Networks (`v4.py`)

- Adds position-wise feed-forward networks
- Increases model capacity and expressiveness

### 5. Complete GPT (`v5.py`)

- Layer normalization for training stability
- Dropout for regularization
- Residual connections
- Full transformer block implementation

## 🔬 Key Concepts Demonstrated

- **Self-attention mechanism** - How tokens attend to each other
- **Positional encoding** - How the model understands sequence order
- **Causal masking** - Preventing the model from "cheating" by looking ahead
- **Token embeddings** - Converting characters to dense vectors
- **Autoregressive generation** - Generating text one token at a time

## 🎓 Educational Value

This project is perfect for:

- Understanding transformer architecture from first principles
- Learning PyTorch implementation details
- Experimenting with hyperparameters
- Seeing how simple components build up to powerful models

## 🛠️ Customization

You can easily modify the model for your own experiments:

1. **Change the dataset**: Replace `input.txt` with your own text
2. **Adjust model size**: Modify `n_embd`, `n_head`, `n_layer`
3. **Experiment with hyperparameters**: Try different learning rates, batch sizes
4. **Add features**: Implement techniques like gradient clipping, learning rate scheduling

## 📈 Future Improvements

- [ ] Add gradient clipping
- [ ] Implement learning rate scheduling
- [ ] Add model checkpointing
- [ ] Support for different tokenization schemes
- [ ] Evaluation metrics (perplexity, BLEU)
- [ ] Distributed training support

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Submit bug reports
- Propose new features
- Improve documentation
- Add more model variants

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this implementation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Andrej Karpathy for the educational content on transformers
- The PyTorch team for the excellent deep learning framework
- Shakespeare for providing the training data 📚

---

⭐ **Star this repo if you found it helpful!** ⭐

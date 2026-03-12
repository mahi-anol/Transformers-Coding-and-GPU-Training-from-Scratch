# Transformer from Scratch: English to French Translation

A complete implementation of the Transformer architecture for neural machine translation, built entirely from scratch using PyTorch. This project translates English text to French using the opus_books dataset.

## Why I Built This

I started this project to move beyond just using pre-trained models and actually understand how Transformers work at a fundamental level. Reading the original "Attention is All You Need" paper was valuable, but seeing the architecture come to life through code really solidified my understanding. Each component, from multi-head attention to positional encoding, had specific design choices and trade-offs that became clear during implementation.

The goal wasn't to create a production-grade system but to grasp the mechanics. How the encoder and decoder communicate through attention, why causal masking matters in autoregressive generation, and how all the pieces fit together. This hands-on approach revealed details that explanations alone couldn't convey.

## Project Structure

```
├── src/
│   ├── data_ingesion/          # Dataset loading and preprocessing
│   │   └── ingestion.py
│   ├── data_preprocessing/     # Tokenizer building
│   │   └── build_tokenizer.py
│   ├── modeling_architechture/ # Core architecture components
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── multihead_attention_block.py
│   │   ├── feed_forward_layer.py
│   │   ├── layer_normalization_block.py
│   │   ├── positional_embedding_layer.py
│   │   ├── input_embedding_layer.py
│   │   ├── skip_connection.py
│   │   ├── final_projection_layer.py
│   │   └── transformers.py      # Main model class
│   ├── training/               # Training pipeline
│   │   └── trainer.py
│   └── inference/              # Inference utilities
│       └── inference.py
└── notebooks/
    ├── Final_Exepriement.ipynb # Full working example
    ├── Experiment.ipynb
    └── temp-exp.ipynb
```

## Architecture Overview

The model implements the standard Transformer encoder-decoder architecture:

**Encoder:** Processes the input English sentence through stacked layers of multi-head self-attention and feed-forward networks, producing a contextual representation.

**Decoder:** Generates the French translation autoregressively, attending to the encoder output and previous tokens via masked self-attention.

**Key Components:**
- Multi-head attention for learning different representation subspaces
- Positional encoding to provide sequence order information
- Feed-forward layers for non-linear transformations
- Layer normalization and skip connections for training stability
- Causal masking to enforce autoregressive generation

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch (CPU or GPU)
- CUDA 11.8+ (optional, for GPU training)

### Step 1: Create Virtual Environment

```bash
python -m venv trans-env
trans-env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets tokenizers tqdm
```

For CPU-only (if no GPU available):
```bash
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Training from Scratch

### Data Preparation

The training pipeline automatically handles data loading and preprocessing:

1. Downloads the opus_books English-French dataset
2. Splits into train/validation/test (80/10/10)
3. Builds word-level tokenizers for both languages
4. Pads sequences and generates attention masks

### Running Training

```python
from src.data_ingesion.ingestion import data_ingestion
from src.data_preprocessing.build_tokenizer import build_tokenizer
from src.modeling_architechture.transformers import transformers
from src.training.trainer import train
import torch

# Load data
train_data, val_data, test_data = data_ingestion()

# Build tokenizers
tokenizer_en = build_tokenizer({'tokenizer_file': 'tokenizer_en.json'}, train_data, 'en')
tokenizer_fr = build_tokenizer({'tokenizer_file': 'tokenizer_fr.json'}, train_data, 'fr')

# Create dataloaders (see notebooks for full dataset/dataloader setup)
# train_loader, val_loader = ...

# Initialize model
model_config = {
    "enc_max_seq_len": 432,
    "dec_max_seq_len": 432,
    "enc_cfg": {
        "emb_dim": 256,
        "no_of_enc_blk": 3,
        "n_heads": 4,
        "pos_emb_dropout": 0.1,
        "mha_dropout": 0.1,
        "expand_dim": 1024,
        "ff_dropout": 0.1,
        "sk_dropout": 0.1
    },
    "dec_cfg": {
        "emb_dim": 256,
        "no_of_dec_blk": 3,
        "pos_emb_dropout": 0.1,
        "n_heads": 4,
        "mha_dropout": 0.1,
        "expand_dim": 1024,
        "ff_dropout": 0.1,
        "sk_dropout": 0.1
    }
}

tokenizer_config = {
    'en_vocab_size': tokenizer_en.get_vocab_size(),
    'fr_vocab_size': tokenizer_fr.get_vocab_size(),
    'fr_pad_token_id': tokenizer_fr.token_to_id('[PAD]')
}

model = transformers(model_config, tokenizer_config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Train
training_config = {
    'epochs': 3,
    'optimizer_lr': 1e-4,
    'optimizer_eps': 1e-8,
    'label_smoothing': 0.1
}

history = train(model, train_loader, val_loader, training_config, tokenizer_config, device)
```

The training process outputs loss and accuracy metrics for each epoch. Full working example available in `Final_Exepriement.ipynb`.

## Inference

### Single Sentence Translation

```python
from src.inference.inference import generate

english_text = "I am who I am"
french_translation = generate(
    model, 
    english_text, 
    tokenizer_en, 
    tokenizer_fr,
    max_seq_len=432,
    device=device
)

print(f"English: {english_text}")
print(f"French: {french_translation}")
```

### Batch Translation

```python
from src.inference.inference import generate_batch

english_texts = [
    "I am who I am",
    "Who are you?",
    "How are you?"
]

french_translations = generate_batch(
    model,
    english_texts,
    tokenizer_en,
    tokenizer_fr,
    max_seq_len=432,
    device=device
)

for eng, fr in zip(english_texts, french_translations):
    print(f"{eng} -> {fr}")
```

## Key Implementation Details

**Tokenization:** Uses the Huggingface tokenizers library with word-level tokenization and special tokens for padding, start-of-sequence, and end-of-sequence.

**Batch Processing:** Custom collate function handles variable-length sequences by padding to the maximum length within each batch, improving GPU memory efficiency.

**Attention Masking:** Encoder mask prevents attention to padding tokens. Decoder mask combines padding mask with causal masking to ensure positions only attend to previous tokens.

**Training:** Uses Adam optimizer with label smoothing to prevent overconfidence. Cross-entropy loss ignores padding tokens during gradient computation.

**Generation:** Autoregressive decoding where each token is generated based on encoder output and previously generated tokens, stopping when EOS token is produced.

## Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Nikita and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  journal={arXiv preprint arXiv:1706.03762},
  year={2017}
}
```

The original paper introduces the Transformer architecture and is foundational for this implementation. Available at: https://arxiv.org/abs/1706.03762

## Dataset

This project uses the opus_books dataset from Huggingface, specifically the English-French translation pair:

```bibtex
@inproceedings{tiedemann2012parallel,
  title={Parallel Data, Tools and Interfaces in OPUS},
  author={Tiedemann, J},
  booktitle={LREC},
  pages={2214--2218},
  year={2012}
}
```

## Notes

The model is intentionally kept relatively small (256-dim embeddings, 3 encoder/decoder layers) for faster training and easier understanding. Scaling to larger dimensions and more layers would improve translation quality as per the original paper's findings.

This is a learning project rather than a production system. For actual deployment, consider using pre-trained models like mBART, mT5, or fine-tuned versions of larger Transformers which have stronger performance guarantees.

## References

- Vaswani et al. "Attention is All You Need" (2017)
- Helsinki-NLP opus_books dataset
- PyTorch documentation
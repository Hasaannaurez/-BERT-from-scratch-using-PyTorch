# Implementation of BERT:
**Flow:**  
1) BERT Data Preparation – `data_prep.py`  
2) BERT Model – `model.py`  
3) BERT Pretraining Script – `train.py`  

## Setup
Install all required dependencies before running:

`pip install -r requirements.txt`
------------------------------------------------------------------------------------------------------------------------------------------------

# BERT Data Preparation - data_prep.py

This file prepares WikiText-2 for BERT-style pretraining by generating masked tokens and next sentence prediction (NSP) pairs.

## Workflow

### 1. Load and Split
`load_and_split_wikitext_2()` loads the WikiText-2 dataset and splits each article into sentences using NLTK.

### 2. Build Sentence Pairs
`build_sentence_pairs()` forms:
- Positive pairs: consecutive sentences  
- Negative pairs: random sentences from other documents  
Each pair gets an NSP label (1 for next, 0 for random).

### 3. Mask Tokens
`create_masked_in_labels()` randomly masks about 15% of tokens:
- 80% → `[MASK]`  
- 10% → random token  
- 10% → unchanged
Labels store the original tokens for the MLM task.

### 4. Dataset
`BertPretrainingDataset` tokenizes sentence pairs, applies masking, and returns tensors for:
- `input_ids`, `token_type_ids`, `attention_mask`  
- `mlm_labels`, `nsp_label`

### 5. Test
Running the file (`data_prep.py`) executes `small_test()` to print a few masked examples.


**Flow of 'data_prep.py`:**  
WikiText-2 → Sentences → Sentence pairs → Tokenized → Masked → Ready for BERT pretraining
------------------------------------------------------------------------------------------------------------------------------------------------

# BERT Model - model.py
This file implements a BERT architecture for pretraining tasks — Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

## Components

### 1. Weight Initialization
`init_weights()` initializes all Linear and Embedding layers with a normal distribution and zeros for biases, ensuring stable training.

### 2. Embeddings
`BertEmbeddings` combines word, position, and token type embeddings, followed by LayerNorm and dropout to produce contextual input representations.

### 3. Multi-Head Self-Attention
`MultiHeadSelfAttention` projects input into query, key, and value tensors, splits them across multiple heads, applies scaled dot-product attention, and merges the results.

### 4. Feed Forward
`FeedForward` applies a two-layer MLP with GELU activation and dropout for non-linear transformation.

### 5. Transformer Encoder
`TransformerEncoderLayer` stacks attention and feed-forward blocks with residual connections and layer normalization.  
`TransformerEncoder` repeats this structure across multiple layers.

### 6. Pretraining Model
`SimpleBertForPreTraining` combines:
- Embeddings and encoder for contextual representations  
- A **pooler** for sentence-level output (NSP)  
- A **MLM head** for token prediction  

It outputs token-level prediction scores and NSP classification logits.

### 7. Test
Running the file (`model.py`) executes `_test_small_forward()` to verify output dimensions for both MLM and NSP heads.

**Flow of 'model.py`:**  
Input tokens → Embeddings → Encoder layers → MLM & NSP heads → Pretraining outputs

------------------------------------------------------------------------------------------------------------------------------------------------


# BERT Pretraining Script - train.py

This file trains the **SimpleBertForPreTraining** model on the **WikiText-2** dataset for Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

## Overview

The script integrates the data pipeline from `data_prep.py` and the model defined in `model.py` to train a compact BERT architecture end-to-end.

### 1. Setup
- Loads the `bert-base-uncased` tokenizer
- Prepares WikiText-2 using `load_and_split_wikitext_2()` and `build_sentence_pairs()`  
- Initializes model, optimizer, and loss functions (CrossEntropy for both MLM and NSP)

### 2. Training
- For each epoch, the model:
  - Performs forward passes on tokenized and masked inputs  
  - Computes MLM and NSP losses  
  - Backpropagates and updates weights with AdamW  
- Progress and average loss are displayed using `tqdm`

### 3. CPU vs GPU
When running locally (CPU), training is limited to `pairs[:1000]` for quick testing.  
On GPU (Google Colab), I trained the full WikiText-2 dataset for complete pretraining.

### 4. Sample Training Results (Full WikiText-2 on Tesla T4)
| Epoch | Train Loss | Val MLM Loss | Val NSP Loss | Val Total | MLM Acc | NSP Acc |
|:------|:------------|:--------------|:--------------|:-----------|:---------|:---------|
| 1 | 7.2328 | 5.5439 | 0.5770 | 6.1209 | 24.16% | 65.62% |
| 2 | 6.4505 | 4.9421 | 0.5631 | 5.5052 | 33.92% | 67.50% |
| 3 | 6.0936 | 4.7954 | 0.5627 | 5.3581 | 35.90% | 67.96% |


**Flow of 'train.py`:**  
WikiText-2 → Sentence pairs → Masked tokens → Model training → MLM & NSP loss optimization
# Grapheme-to-Phoneme (G2P) Model Documentation

This document serves as both the installation guide for using the Grapheme-to-Phoneme (G2P) model and a full walkthrough of its code and development process.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
2. [Introduction](#introduction)
   - [What is G2P Conversion?](#what-is-g2p-conversion)
   - [What are the Model Options?](#what-are-the-model-options)
   - [Why Use an Attention-Based Seq2Seq Model?](#why-use-an-attention-based-seq2seq-model)
3. [Model Overview](#model-overview)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architecture](#model-architecture)
4. [G2P Model Code Design](#g2p-model-code-design)
   - [Importing Libraries](#importing-libraries)
   - [Defining the G2P Class](#defining-the-g2p-class)
5. [G2P Model Training Code Design](#g2p-model-training-code-design)
   - [Importing Libraries](#importing-libraries-1)
   - [Preparing the Dataset](#preparing-the-dataset)
   - [Training Loop](#training-loop)
6. [Evaluating Model Weights](#evaluating-model-weights)
7. [Model Optimizations and Comparisons](#model-optimizations-and-comparisons)
8. [References](#references)

## Installation and Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA and Tensor Cores (for training)
  - If your GPU does not have CUDA Cores, the code will automatically train on your CPU.
  - If your GPU does not have Tensor Cores, mixed precision will still be used, but performance won't be as good.

### Installation Steps

**1. Set Up a Virtual Environment (Optional)**

- Create a virtual environment:
  ```
  python -m venv venv
  ```
- Activate the virtual environment:
  ```
  .\venv\Scripts\activate      # Windows
  source venv/bin/activate     # Linux/macOS
  ```

**2. Install Required Packages**

- First, if you are training the model and want CUDA support, install PyTorch using:
  ```
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
  ```
  - Note: You do not need NVIDIA CUDA Toolkit installed to use PyTorch with CUDA support.
- Second, install the remaining packages using:
  ```
  pip install spacy editdistance          # If torch was installed for CUDA support
  pip install spacy editdistance torch    # If CUDA support will not be used
  ```
- Third, download the spaCy English Model using:
  ```
  python -m spacy download en_core_web_sm
  ```

### Model Usage

To use the model, either use the `generate-phonetics.py` and change the `text` variable or create a different file and import `G2P` from `g2p.py` and pass in the path to the model weights (pre-trained weights are saved in `g2p-assets/model-checkpoint.pt`).

## Introduction

### What is G2P Conversion?

Grapheme-to-Phoneme (G2P) conversion is the process of converting written text (graphemes) into their corresponding pronunciations (phonemes). G2P models are used for text-to-speech (TTS) systems, speech recognition, and [Gibberle](https://github.com/Pv3-Labs/gibberle)!

**Example**:

- **Word**: "testing"
- **Pronunciation**: T EH1 S T IH0 NG

### What are the Model Options?

When building a G2P Model, there are several different model architectures we can consider with their own advantages and disadvantages. Listed below are some common approaches:

**1. Rule-Based Models**

Rule-based models are the simplest form of G2P conversion. These models utilize a set of rules to map graphemes to phonemes. For example, a rule-based system could define "ph" as being mapped to the phoneme "F."

- **Advantages:**
  - No training data is required.
  - Very simple to understand and implement.
  - Works well for languages with consistent spelling rules.
- **Disadvantages:**
  - All rule-based mappings must be manually defined.
  - Hard to handle all possible exceptions and words.
  - Difficult to generalize for irregular or complex languages (e.g., English).

**2. Seq2Seq Models**

A Sequence-to-Sequence (Seq2Seq) model is a type of Recurrent Neural Network (RNN) that uses an encoder-decoder architecture to map graphemes to phonemes in a step-by-step process. The encoder transforms the entire input sequence into a fixed-length vector called a context vector, which the decoder then uses to generate the output sequence.

- **Advantages:**
  - Handles variable input/output lengths.
  - Learns mappings from data instead of relying on predefined rules.
- **Disadvantages:**
  - A fixed-length context vector can limit the model's ability to handle long input sequences.
  - Model must be trained on data, which is computationally intensive.

**3. Attention-based Seq2Seq Models**

Attention-based models extend the Seq2Seq architecture by allowing the decoder to "focus" on specific parts of the input sequence during each decoding step. Instead of using a fixed-length context vector, the attention mechanism dynamically computes a weighted sum of the encoder's hidden states, providing more information for the decoder.

- **Advantages:**
  - Better handling of long input sequences.
  - Typically has higher accuracy than the previous models.
- **Disadvantages:**
  - Computationally more expensive than the previous models.
  - Usually requires more training data and tuning of hyperparameters than a regular Seq2Seq model for optimal weights.

**4. Transformer Models**

Transformer models are the most complex, but powerful, form of G2P conversion. Transformer models use self-attention mechanisms to process the entire input sequence at once, instead of an encoder-decoder architecture.

- **Advantages:**
  - Better handling of long input sequences than Seq2Seq models.
  - Able to generalize irregular or complex languages.
  - Typically has higher accuracy than the previous models.
- **Disadvantages:**
  - Requires large amounts of training data.
  - Computationally more expensive than the previous models.
  - Much more complex and difficult to implement than the previous models.

### Why Use an Attention-Based Seq2Seq Model?

Due to the requirements of Gibberle, a rule-based model would be inconvenient due to the complexity of English. Additionally, a transformer model would also not be a good fit due to the complexity of implementation. Therefore, we chose to move forward with a Seq2Seq model. Specifically, we chose to utilize the attention-based Seq2Seq model because of the variability of input length we'd need to create our gibberish.

## Model Overview

### Data Preprocessing

Before passing data into the model, we need to first:

- **Normalize Text:** Remove diacritics (e.g., accent marks), convert to lowercase, and remove all non-alphabetic characters except apostrophes.
- **Tokenize Input:** After the text has been normalized, we need to tokenize the input so it can be converted into graphemes that can be mapped to indices for use by the model.

### Model Architecture

As described in [What are the Model Options?](#what-are-the-model-options), the Seq2Seq model uses an encoder-decoder architecture with an attention mechanism. Additionally, Gated Recurrent Units (GRU) were chosen to implement the RNN over Long Short-Term Memory (LSTM) due to their simplicity, faster training time, and ability to achieve similar performance. Below is a diagram showcasing the model achitecture by Arthur Suilin.

![image](g2p-assets/encoder-decoder.png)

#### Embeddings

**Purpose:** Convert tokens (graphemes and phonemes) into vector representations.

- **Grapheme Embedding Layer:** Maps input characters to embedding vectors.
- **Phoneme Embedding Layer:** Maps output phonemes to embedding vectors.

**Implementation:**

```python
self.enc_emb = nn.Embedding(len(self.graphemes), embedding_dim)
self.dec_emb = nn.Embedding(len(self.phonemes), embedding_dim)
```

#### Encoder

**Purpose:** Encode the input sequence of graphemes into a context vector.

- **GRU Layer:** Processes the embedded input sequence and captures sequential dependencies.

**Implementation:**

```python
self.enc_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
```

#### Decoder

**Purpose:** Generates the output sequence of phonemes, one token at a time.

- **GRU Layer:** Takes the previous output and context vector to produce the next phoneme.

**Implementation:**

```python
self.dec_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
```

#### Attention Mechanism

**Purpose:** Allows the decoder to focus on different parts of the input sequence at each step.

- **Computes:** Alignment scores between the decoder hidden state and encoder outputs.
- **Generates:** A context vector as a weighted sum of encoder outputs.

**Implementation:**

```python
self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
self.attn_combine = nn.Linear(hidden_size + embedding_dim, embedding_dim)
```

## G2P Model Code Design (`g2p.py`)

### Importing Libraries

```python
import os  # For handling file paths and checking file existence
import re  # For regular expressions, used in text normalization
import unicodedata  # For handling Unicode normalization of text

import spacy  # For tokenizing text into words using the 'en_core_web_sm' model
import torch  # Core PyTorch library for tensor operations
import torch.nn as nn  # For defining neural network layers
import torch.nn.functional as F  # For additional functions like softmax

nlp = spacy.load('en_core_web_sm')  # For utilizing spaCy tokenization
```

### Defining the G2P Class

The `G2P` class inherits from `nn.Module`, which is the base class for all neural network modules in PyTorch.

```python
class G2P(nn.Module):
    """
    Grapheme-to-Phoneme (G2P) Model using an encoder-decoder architecture with attention.

    This model converts a sequence of graphemes (characters) into a sequence of phonemes.
    It uses embedding layers for both graphemes and phonemes, GRU layers for encoding and decoding,
    and an attention mechanism to focus on relevant parts of the input during decoding.
    """
```

#### Initalization

In the `__init__` method, we initialize all the components of the model (credit to [Kyubyong](https://github.com/Kyubyong/g2p) for the original structure).

```python
def __init__(self, checkpoint_path=None):
    super(G2P, self).__init__()
    # Define graphemes and phonemes
    # Create mappings
    # Define hyperparameters
    # Initialize layers
    # Load variables if checkpoint is provided
```

**Defining Graphemes and Phonemes:**

We define lists of graphemes and phonemes, including special tokens:

```python
self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")

self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
    'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',
    'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0',
    'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
    'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0',
    'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]
```

- **Special Tokens:**
  - `"<pad>"`: Padding token used for sequences of different lengths.
  - `"<unk>"`: Unknown token for graphemes or phonemes not in the vocabulary.
  - `"<s>"`: Start-of-sequence token for the decoder.
  - `"</s>"`: End-of-sequence token indicating the end of a sequence.

**Creating Mappings:**

We create dictionaries to map tokens to indices and vice versa to allow us to convert between token strings and their corresponding numerical indices required by embedding layers:

```python
self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
```

**Defining Hyperparameters:**

```python
embedding_dim = 500
hidden_size = 500
dropout_rate = 0.6
```

- `embedding_dim`: Size of the embedding vectors.
- `hidden_size`: Number of features in the hidden state of GRU layers.
- `dropout_rate`: Probability of zeroing elements during training to prevent overfitting.

**Initializing Layers:**

In addition to the layers in [Model Architecture](#model-architecture), we also have:

- **Output Layer**:

  ```python
  self.fc = nn.Linear(hidden_size, len(self.phonemes))
  ```

  - **Purpose:** Maps the decoder GRU output to phoneme probabilities.

- **Dropout Layer**:
  ```python
  self.dropout = nn.Dropout(p=dropout_rate)
  ```
  - **Purpose**: Regularization to prevent overfitting.

**Loading Variables:**

Model weights can be optionally loaded from a checkpoint:

```python
self.load_variables(checkpoint_path)

def load_variables(self, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found or not provided. Using random weights.")
```

#### Foward Method (Encoder)

The `forward` method processes the input grapheme sequence through the encoder.

```python
def forward(self, x):
    """
    Forward pass through the encoder.

    Args:
        x (Tensor): Input tensor of grapheme indices with shape (batch_size, seq_length).

    Returns:
        Tuple[Tensor, Tensor]: Encoder outputs and the final hidden state.
    """
    # Embed the input graphemes and apply dropout
    embedded = self.dropout(self.enc_emb(x))

    # Pass the embeddings through the encoder GRU
    encoder_outputs, hidden = self.enc_gru(embedded)

    return encoder_outputs, hidden
```

- `encoder_outputs`: All hidden states from the GRU for each step.
- `hidden`: The final hidden state of the encoder GRU.

#### Decode Step Method

The `decode_step` method performs a single decoding step with attention.

```python
def decode_step(self, decoder_input, decoder_hidden, encoder_outputs):
    """
    Perform a single decoding step with attention.

    Args:
        decoder_input (Tensor): Current input to the decoder (phoneme indices) with shape (batch_size, 1).
        decoder_hidden (Tensor): Current hidden state of the decoder with shape (1, batch_size, hidden_size).
        encoder_outputs (Tensor): Outputs from the encoder with shape (batch_size, seq_length, hidden_size).

    Returns:
        Tuple[Tensor, Tensor]: Output probabilities for the next phoneme and updated hidden state.
    """
    # Embed the decoder input and apply dropout
    embedded = self.dropout(self.dec_emb(decoder_input))

    # Compute attention weights using batch matrix multiplication
    # Transpose decoder_hidden to (batch_size, 1, hidden_size) for bmm
    attn_weights = torch.bmm(decoder_hidden.transpose(0, 1), encoder_outputs.transpose(1, 2))

    # Apply softmax to get normalized attention weights
    attn_weights = F.softmax(attn_weights, dim=2)

    # Compute the context vector as a weighted sum of encoder outputs
    context = torch.bmm(attn_weights, encoder_outputs)

    # Concatenate embedded input and context vector
    rnn_input = torch.cat((embedded, context), dim=2)

    # Pass the concatenated vector through the decoder GRU
    output, decoder_hidden = self.dec_gru(rnn_input, decoder_hidden)

    # Pass the GRU output through the fully connected layer to get phoneme probabilities
    output = self.fc(output.squeeze(1))

    return output, decoder_hidden
```

#### Predict Method

The `predict` method generates the phoneme sequence for a given word using beam search (credit to [Fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme/) for the idea).

```python
def predict(self, word, beam_width=3):
    """
    Predict the phoneme sequence for a given word using beam search.

    Args:
        word (str): The input word to convert to phonemes.
        beam_width (int): The number of beams to keep during search.

    Returns:
        List[str]: The predicted sequence of phonemes.
    """
    self.eval()  # Set the model to evaluation mode

    # Convert the word into a list of characters and append the end-of-sequence token
    chars = list(word) + ["</s>"]

    # Map characters to their corresponding indices, using <unk> for unknown characters
    x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]

    # Convert to a tensor and add batch dimension
    x = torch.tensor(x).unsqueeze(0).to(next(self.parameters()).device)

    # Pass the input through the encoder
    encoder_outputs, hidden = self.forward(x)

    # Initialize decoder hidden state with encoder's final hidden state
    decoder_hidden = hidden  # Shape: (1, batch_size, hidden_size)

    # Initialize the decoder input with the start-of-sequence token
    decoder_input = torch.tensor([[self.p2idx["<s>"]]], device=x.device)  # Shape: (1,1)

    max_length = 50  # Maximum length for phoneme sequence

    # Initialize beams with a tuple of (score, decoder_input_sequence, decoder_hidden_state)
    beams = [(0, decoder_input, decoder_hidden)]
    completed_beams = []

    # Iterate up to the maximum length
    for _ in range(max_length):
        new_beams = []
        for score, decoder_input_seq, decoder_hidden in beams:
            # Get the last token in the decoder input sequence
            last_token = decoder_input_seq[:, -1:]

            # If the last token is end-of-sequence, add to completed beams
            if last_token.item() == self.p2idx["</s>"]:
                completed_beams.append((score, decoder_input_seq))
                continue

            # Perform a decoding step to get output logits and new hidden state
            output, decoder_hidden_new = self.decode_step(last_token, decoder_hidden, encoder_outputs)

            # Convert output logits to log probabilities
            log_probs = torch.log_softmax(output, dim=1)

            # Get the top 'beam_width' log probabilities and their indices
            top_log_probs, top_indices = log_probs.topk(beam_width)

            # Iterate through the top predictions to create new beams
            for i in range(beam_width):
                new_score = score + top_log_probs[0, i].item()  # Accumulate log probabilities
                new_decoder_input_seq = torch.cat([decoder_input_seq, top_indices[:, i:i+1]], dim=1)
                new_beams.append((new_score, new_decoder_input_seq, decoder_hidden_new))

        # Sort the new beams based on their scores in descending order and keep top 'beam_width'
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # If all beams have completed, stop the search
        if all(decoder_input_seq[0, -1].item() == self.p2idx["</s>"] for _, decoder_input_seq, _ in beams):
            break

    # Add any remaining beams to completed beams
    completed_beams.extend(beams)

    # Select the beam with the highest score
    best_beam = max(completed_beams, key=lambda x: x[0])

    # Extract phoneme indices from the best beam, excluding the start token
    output_indices = best_beam[1].squeeze().tolist()[1:]

    # Truncate at end-of-sequence if present
    if self.p2idx["</s>"] in output_indices:
        end_idx = output_indices.index(self.p2idx["</s>"])
        output_indices = output_indices[:end_idx]

    # Convert indices back to phoneme strings
    preds = [self.idx2p[idx] for idx in output_indices]

    return preds
```

**Key Steps:**

1. **Preprocessing**: Convert the input word to grapheme indices, adding the end-of-sequence token.

2. **Encoder Forward Pass**: Obtain encoder outputs and hidden state.

3. **Initialize Beam Search**: Start with initial beams containing the start token and initial decoder hidden state.

4. **Beam Search Loop**:

   - For each beam, perform a decoding step.
   - Compute log probabilities for possible next phonemes.
   - Create new beams by extending current beams with top candidates.
   - Prune beams to keep only the top `beam_width` beams based on accumulated scores.

5. **Termination Conditions**:

   - Stop if all beams have generated the end-of-sequence token.
   - Stop after a maximum length to prevent infinite loops.

6. **Select Best Beam**: Choose the beam with the highest accumulated score.

7. **Postprocessing**: Convert indices back to phoneme tokens, excluding special tokens.

**Beam Search Advantages:**

- **Explores Multiple Paths**: Unlike greedy decoding, beam search keeps multiple hypotheses, increasing the chance of finding the best sequence.
- **Balances Efficiency and Accuracy**: Beam width controls the trade-off between computational cost and performance.

#### Text Normalization

The `normalize_text` method handles the preprocessing of input text (credit to [Kyubyong](https://github.com/Kyubyong/g2p) for original structure).

```python
def normalize_text(self, text):
    """
    Normalize input text by removing diacritics, converting to lowercase,
    and removing non-alphabetic characters except apostrophes.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    # Remove diacritics using Unicode normalization
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    # Convert to lowercase
    text = text.lower()
    # Remove characters that are not a-z, space, or apostrophe
    text = re.sub(r"[^ a-z']", "", text)
    return text
```

#### Call Method

Overrides the `__call__` method to process input text and return phonemes (credit to [Kyubyong](https://github.com/Kyubyong/g2p) for original structure).

```python
def __call__(self, text):
    """
    Convert input text to a sequence of phonemes.

    Args:
        text (str): The input text to convert.

    Returns:
        List[str]: The resulting phoneme sequence.
    """
    # Normalize the input text
    text = self.normalize_text(text)

    # Tokenize the text using spaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]

    prons = []  # List to hold phoneme sequences
    for word in tokens:
        if not re.search("[a-z]", word):
            # If the token is not alphabetic, treat it as a single phoneme
            pron = [word]
        else:
            # Predict phonemes for the word
            pron = self.predict(word.lower())
        prons.extend(pron)
        prons.append(" ")  # Add space as a phoneme separator
    return prons[:-1]  # Remove the trailing space
```

## G2P Model Training Code Design (`train-g2p.py`)

### Importing Libraries

```python
import os  # For handling file paths and checking file existence
import random  # For shuffling the dataset to ensure randomness in training
import sys  # For adding directories to the Python path

import torch  # Core PyTorch library for tensor operations
import torch.nn as nn  # For defining loss functions and neural network layers
from torch.amp import (  # For mixed precision training to speed up training on GPUs
    GradScaler, autocast)
from torch.nn.utils.rnn import \
    pad_sequence  # For padding variable-length sequences
from torch.utils.data import (  # For batching and managing training data
    DataLoader, Dataset)
from tqdm import tqdm  # For displaying progress bars during training

# Add parent directory to system path to import G2P model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g2p import G2P  # Import the G2P model class for training
```

### Preparing the Dataset

To train the model, the **Carnegie Mellon University Pronouncing Dictionary (CMUdict)** was used. Since the raw `cmudict.dict` file contains structure like:

```
batignolles B AE2 T IH0 N Y OW1 L AH0 S
batik B AH0 T IY1 K
batiks B AE1 T IH0 K S
batiks(2) B AH0 T IY1 K S
batista B AH0 T IH1 S T AA0
batista's B AH0 T IH1 S T AA0 Z
batiste B AH0 T IH1 S T AH0
batley B AE1 T L IY0
```

We must load and parse the data for utilization in our model.

**Loading Data:**

The `load_cmudict_data` method loads and parses `cmudict.dict`.

```python
def load_cmudict_data(cmudict_file):
    """
    Load and parse the CMU Pronouncing Dictionary data.

    Args:
        cmudict_file (str): Path to the CMUdict file.

    Returns:
        List[dict]: A list of dictionaries containing word and phonemes.
    """
    data = []
    with open(cmudict_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            word = parts[0].split('(')[0]  # Remove any parenthetical indices for alternate pronunciations
            phonemes = parts[1:]
            data.append({'word': word, 'phonemes': phonemes})
    return data
```

**Custom Dataset Class:**

To utilize `DataLoader`, we must create a custom `Dataset` subclass to handle data loading.

```python
class CMUdictDataset(Dataset):
    """
    Custom Dataset for the CMU Pronouncing Dictionary.

    Each item in the dataset consists of input grapheme indices and target phoneme indices.
    """
    def __init__(self, data, g2p_model):
        self.data = data
        self.g2p_model = g2p_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        word = sample['word']
        phonemes = sample['phonemes']

        # Convert the word to lowercase and append end-of-sequence token
        graphemes = list(word.lower()) + ["</s>"]

        # Map graphemes to their corresponding indices, using <unk> for unknown graphemes
        input_ids = [self.g2p_model.g2idx.get(char, self.g2p_model.g2idx["<unk>"]) for char in graphemes]

        # Prepare target phoneme indices with start and end tokens
        labels = [self.g2p_model.p2idx["<s>"]] + \
                 [self.g2p_model.p2idx.get(ph, self.g2p_model.p2idx["<unk>"]) for ph in phonemes] + \
                 [self.g2p_model.p2idx["</s>"]]

        return {'input_ids': input_ids, 'labels': labels}
```

**Collate Function:**

Once again, to utilize `DataLoader`, we must create a `collate_batch` method that pads input and label sequences to have a uniform length.

```python
def collate_batch(batch):
    """
    Collate function to be used with DataLoader for batching.

    Pads input and label sequences to the maximum length in the batch.

    Args:
        batch (List[dict]): A list of samples from the dataset.

    Returns:
        dict: A dictionary containing padded input_ids and labels tensors.
    """
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    # Pad sequences to the same length using <pad> token
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=g2p_model.g2idx["<pad>"])
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=g2p_model.p2idx["<pad>"])

    return {'input_ids': input_ids_padded, 'labels': labels_padded}
```

### Training Loop

The model was trained using a standard training loop with some additional optimizations for performance and fitting.

**Loss Function and Optimizer:**

- **Loss Function**: Cross-Entropy Loss, ignoring the padding index.

  ```python
  criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
  ```

- **Optimizer:** AdamW optimizer with weight decay (prevent overfitting).

  ```python
  optimizer = torch.optim.AdamW(g2p_model.parameters(), lr=0.0005, weight_decay=1e-3)
  ```

- **Learning Rate Scheduler**: Reduces the learning rate when validation loss plateaus (prevent overfitting).
  ```python
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
  ```

**Implementing Optimizations:**

- **Gradient Clipping**: Prevents exploding gradients to help with convergence (credit to [Fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme/) for the idea).

  ```python
  torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)
  ```

- **Learning Rate Scheduling**: Adjusts learning rate based on validation loss (credit to [Fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme/) for the idea).

  ```python
  scheduler.step(val_loss)
  ```

- **Early Stopping:** Stops training if no improvement in validation loss after `patience` epochs (credit to [Fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme/) for the idea).

  ```python
  if epochs_no_improve >= patience:
    print("Stopping early due to no validation loss improvement.")
    break
  ```

- **Mixed Precision Training**: Speeds up training on GPUs with CUDA (and Tensor) Cores.
  ```python
  scaler = GradScaler()
  with autocast(device_type='cuda'):
      # Forward pass and loss computation
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

**Full Training Loop:**

```python
for epoch in range(num_epochs):
    g2p_model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        # Mixed Precision Training
        if cuda_available:
            with autocast(device_type='cuda'):
                # Forward pass
                encoder_outputs, encoder_hidden = g2p_model.forward(input_ids)
                # Prepare decoder inputs and targets
                decoder_input = labels[:, :-1]
                decoder_targets = labels[:, 1:]
                decoder_hidden = encoder_hidden
                outputs = []
                # Decode sequence
                for t in range(decoder_input.size(1)):
                    decoder_input_t = decoder_input[:, t].unsqueeze(1)
                    output, decoder_hidden = g2p_model.decode_step(decoder_input_t, decoder_hidden, encoder_outputs)
                    outputs.append(output)
                outputs = torch.stack(outputs, dim=1)
                # Compute loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.reshape(-1))
            # Backpropagation and optimization with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Training
            # Forward pass
            # Similar steps as above without scaling
            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
    # Validation and learning rate adjustment
    val_loss = evaluate(g2p_model, valid_loader)
    scheduler.step(val_loss)
    # Early stopping
```

## Evaluating Model Weights

The following metrics were used to evaluate model weights (credit to [Fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme/) for the idea):

- **Phoneme Error Rate (PER)**: The ratio of incorrectly predicted phonemes.
- **Word Error Rate (WER):** The percentage of words where the predicted phoneme sequence does not match the target.
- **Average Edit (Levenshtein) Distance:** The average number of insertions, deletions, or substitutions required to change the predicted sequence into the target sequence.

The `evaluate_model` method (seen in `evaluate-model.py`) evaluates a given checkpoint path for our model.

```python
def evaluate_model(checkpoint_path):
    model = G2P(checkpoint_path=checkpoint_path)
    model.eval()
    total_phoneme_count = 0
    correct_phoneme_count = 0
    total_edit_distance = 0
    total_sequences = len(test_dataset)
    correct_word_count = 0

    special_tokens = ["<pad>", "<s>", "</s>"]

    with torch.no_grad():
        for i, (word, target_phonemes_list) in enumerate(tqdm(test_dataset, desc="Processing samples")):
            predicted_phonemes = model.predict(word)
            predicted_phonemes_cleaned = [ph for ph in predicted_phonemes if ph not in special_tokens]
            target_phonemes_cleaned = [ph for ph in target_phonemes_list if ph not in special_tokens]

            correct_phoneme_count += sum(1 for pred, target in zip(predicted_phonemes_cleaned, target_phonemes_cleaned) if pred == target)
            total_phoneme_count += len(target_phonemes_cleaned)
            total_edit_distance += editdistance.eval(predicted_phonemes_cleaned, target_phonemes_cleaned)

            if predicted_phonemes_cleaned == target_phonemes_cleaned:
                correct_word_count += 1

    phoneme_level_accuracy = correct_phoneme_count / total_phoneme_count if total_phoneme_count > 0 else 0
    average_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0
    word_level_accuracy = correct_word_count / total_sequences if total_sequences > 0 else 0

    return phoneme_level_accuracy, average_edit_distance, word_level_accuracy
```

## Model Optimizations and Comparisons

**Baseline Model:** First, we train a baseline model without any optimizations:

- No Gradient Clipping
- No Learning Rate Scheduling
- No Early Stopping
- No Mixed Precision Training
- Greedy Decoding Instead of Beam Search

**Optimized Model:** Second, we train a model with optimizations:

- Gradient Clipping
- Learning Rate Scheduling
- Early Stopping
- Mixed Precision Training
- Beam Search Decoding

**Comparisons:**

```
Baseline Model    | Phoneme-level Accuracy: 80.84%, Average Edit Distance: 0.98, Word-level Accuracy: 48.96%
Optimized Model   | Phoneme-level Accuracy: 91.89%, Average Edit Distance: 0.39, Word-level Accuracy: 75.39%
```

As we can see from the comparison, the optimized model performs better (yay!). The optimized model was also able to converge faster, although per-epoch training time was the same. Additionally, the baseline model wasted training time once it had converged since there was no early stopping.

## References

- **Kyubyong's G2P Implementation**: [g2pE: A Simple Python Module for English Grapheme To Phoneme Conversion](https://github.com/Kyubyong/g2p)
- **Fehiepsi's Blog Post**: [How to build a Grapheme-to-Phoneme (G2P) model using PyTorch](https://fehiepsi.github.io/blog/grapheme-to-phoneme/)
- **Arthur Suilin's Seq2Seq Diagram**: [Kaggle Web Traffic Time Series Forecasting](https://github.com/Arturus/kaggle-web-traffic)

## Thanks for Reading!

```
Original Text:
Thanks for reading

Phonetic representation:
TH AE1 NG K S   F AO1 R   R EH1 D IH0 NG
```

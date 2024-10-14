import os  # For file path operations
import re  # For regular expressions
import unicodedata  # For Unicode normalization

import numpy as np  # For numerical computations
import spacy  # For tokenization and POS tagging
import torch
import torch.nn as nn
from datasets import load_dataset  # For loading the dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bars

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')
dirname = os.path.dirname(__file__)

# Function to construct a dictionary of heteronyms
def construct_heteronym_dictionary():
    """
    Constructs a dictionary for heteronyms (words with multiple pronunciations based on context).
    """
    heteronym_file = os.path.join(dirname, 'heteronyms.txt')
    heteronym2features = dict()
    with open(heteronym_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines
            headword, pron1, pron2, pos1 = line.strip().split("|")
            heteronym2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return heteronym2features

# G2p model using PyTorch
class G2p(nn.Module):
    def __init__(self):
        """
        Initializes the G2p model.
        """
        super(G2p, self).__init__()
        # List of graphemes and phonemes
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
            'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
            'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',
            'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0',
            'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
            'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
            'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0',
            'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]

        # Mappings between graphemes/phonemes and their indices
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        # Embedding dimensions and hidden size
        embedding_dim = 128
        hidden_size = 256

        # Encoder embeddings and GRU
        self.enc_emb = nn.Embedding(len(self.graphemes), embedding_dim)
        self.enc_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        # Decoder embeddings and GRU
        self.dec_emb = nn.Embedding(len(self.phonemes), embedding_dim)
        self.dec_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, len(self.phonemes))

        # Heteronym dictionary
        self.heteronym2features = construct_heteronym_dictionary()

    def forward(self, x):
        """
        Forward pass for the encoder.
        """
        embedded = self.enc_emb(x)
        outputs, hidden = self.enc_gru(embedded)
        return hidden

    def predict(self, word):
        """
        Predicts the phoneme sequence for a given word.
        """
        device = next(self.parameters()).device  # Get the device (CPU or GPU)

        # Encode the input word
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = torch.tensor(x).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Encoder forward pass
        encoder_hidden = self.forward(x)

        # Decoder initialization
        decoder_input = torch.tensor([[self.p2idx["<s>"]]]).to(device)  # Start token
        hidden = encoder_hidden

        preds = []
        max_length = 20
        for _ in range(max_length):
            embedded = self.dec_emb(decoder_input)
            output, hidden = self.dec_gru(embedded, hidden)
            logits = self.fc(output.squeeze(1))
            pred = logits.argmax(dim=1)
            pred_idx = pred.item()
            if pred_idx == self.p2idx["</s>"]:
                break
            preds.append(pred_idx)
            decoder_input = pred.unsqueeze(0).unsqueeze(0)
        # Convert indices to phonemes
        preds = [self.idx2p[idx] for idx in preds]
        return preds

    def normalize_text(self, text):
        """
        Normalizes and preprocesses the input text.
        """
        text = ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )
        text = text.lower()
        text = re.sub(r"[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")
        return text

    def __call__(self, text):
        """
        Processes input text and returns the phonetic representation.
        """
        text = self.normalize_text(text)
        doc = nlp(text)
        tokens = [(token.text, token.tag_) for token in doc]

        prons = []
        for word, pos in tokens:
            if not re.search("[a-z]", word):
                pron = [word]
            elif word.lower() in self.heteronym2features:
                pron1, pron2, pos1 = self.heteronym2features[word.lower()]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            else:
                pron = self.predict(word.lower())
            prons.extend(pron)
            prons.append(" ")
        return prons[:-1]  # Remove the last extra space

# Instantiate the model
g2p_model = G2p()

# Load the s3prl/g2p Dataset
dataset = load_dataset("s3prl/g2p")

# Split the dataset into training and validation sets
dataset = dataset['train'].train_test_split(test_size=0.1)
train_dataset = dataset['train']
valid_dataset = dataset['test']

def preprocess_function(examples):
    input_ids = []
    labels = []
    for line in examples['text']:
        # Each line contains the word followed by its phonemes
        # Example: "ABDUCTORS AE0 B D AH1 K T ER0 Z"
        parts = line.strip().split()
        if len(parts) < 2:
            continue  # Skip lines that don't have both word and phonemes
        word = parts[0].lower()
        phonemes = parts[1:]

        # Convert graphemes to indices
        graphemes = list(word) + ["</s>"]
        input_id = [g2p_model.g2idx.get(char, g2p_model.g2idx["<unk>"]) for char in graphemes]
        input_ids.append(input_id)

        # **Include the <s> token at the start of labels**
        label = [g2p_model.p2idx["<s>"]] + [g2p_model.p2idx.get(ph, g2p_model.p2idx["<unk>"]) for ph in phonemes] + [g2p_model.p2idx["</s>"]]
        labels.append(label)

    return {'input_ids': input_ids, 'labels': labels}

# Apply preprocessing to both training and validation sets
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text'])

def collate_batch(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=g2p_model.g2idx["<pad>"])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=g2p_model.p2idx["<pad>"])

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

# Create DataLoaders
train_loader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(tokenized_datasets['test'], batch_size=32, shuffle=False, collate_fn=collate_batch)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
optimizer = torch.optim.Adam(g2p_model.parameters(), lr=0.001)

# Move model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g2p_model.to(device)

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    total_phonemes = 0
    correct_phonemes = 0
    total_words = 0
    correct_words = 0
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_size = input_ids.size(0)

            # Encoder forward pass
            encoder_hidden = model.forward(input_ids)

            # Initialize decoder input (batch of start tokens)
            decoder_input = torch.full((batch_size, 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
            hidden = encoder_hidden
            max_length = labels.size(1)

            # Store predictions and logits
            all_preds = []
            all_logits = []
            for t in range(max_length):
                decoder_embedded = model.dec_emb(decoder_input)
                output, hidden = model.dec_gru(decoder_embedded, hidden)
                logits = model.fc(output.squeeze(1))
                all_logits.append(logits)
                # Get the highest probability token
                top1 = logits.argmax(1, keepdim=True)
                all_preds.append(top1)
                decoder_input = top1  # Use the predicted token as next input

            # Concatenate predictions and logits
            preds = torch.cat(all_preds, dim=1)
            logits = torch.stack(all_logits, dim=1)  # Shape: (batch_size, seq_len, vocab_size)

            # Compute loss
            decoder_target_ids = labels
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_target_ids.reshape(-1))
            total_loss += loss.item()

            # Compute accuracy
            # Phoneme-level accuracy
            mask = decoder_target_ids != g2p_model.p2idx["<pad>"]
            total_phonemes += mask.sum().item()
            correct_phonemes += ((preds == decoder_target_ids) & mask).sum().item()

            # Word-level accuracy
            for i in range(batch_size):
                target_seq = decoder_target_ids[i][mask[i]].tolist()
                pred_seq = preds[i][mask[i]].tolist()
                total_words += 1
                if pred_seq == target_seq:
                    correct_words += 1

    avg_loss = total_loss / len(data_loader)
    phoneme_accuracy = correct_phonemes / total_phonemes
    word_accuracy = correct_words / total_words
    return avg_loss, phoneme_accuracy, word_accuracy

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    g2p_model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        encoder_hidden = g2p_model.forward(input_ids)

        decoder_input_ids = labels[:, :-1]
        decoder_target_ids = labels[:, 1:]

        decoder_embedded = g2p_model.dec_emb(decoder_input_ids)
        hidden = encoder_hidden
        output, _ = g2p_model.dec_gru(decoder_embedded, hidden)
        logits = g2p_model.fc(output)

        # Flatten the logits and targets for computing cross-entropy loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = decoder_target_ids.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    val_loss, phoneme_acc, word_acc = evaluate(g2p_model, valid_loader)
    print(f"Validation Loss: {val_loss:.4f}, Phoneme Accuracy: {phoneme_acc:.4f}, Word Accuracy: {word_acc:.4f}")

# Save the model
checkpoint_file = os.path.join(dirname, 'checkpoint.pt')
torch.save({
    'enc_emb': g2p_model.enc_emb.weight.data,
    'enc_gru_weight_ih_l0': g2p_model.enc_gru.weight_ih_l0.data,
    'enc_gru_weight_hh_l0': g2p_model.enc_gru.weight_hh_l0.data,
    'enc_gru_bias_ih_l0': g2p_model.enc_gru.bias_ih_l0.data,
    'enc_gru_bias_hh_l0': g2p_model.enc_gru.bias_hh_l0.data,

    'dec_emb': g2p_model.dec_emb.weight.data,
    'dec_gru_weight_ih_l0': g2p_model.dec_gru.weight_ih_l0.data,
    'dec_gru_weight_hh_l0': g2p_model.dec_gru.weight_hh_l0.data,
    'dec_gru_bias_ih_l0': g2p_model.dec_gru.bias_ih_l0.data,
    'dec_gru_bias_hh_l0': g2p_model.dec_gru.bias_hh_l0.data,

    'fc_weight': g2p_model.fc.weight.data,
    'fc_bias': g2p_model.fc.bias.data,
}, checkpoint_file)

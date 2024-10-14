# g2p model for converting graphemes to phonemes
# Inspired from https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

import os  # For file path operations
import re  # For normalizing text
import unicodedata  # For Unicode normalization

import spacy  # For tokenization and POS tagging
import torch  # For model functionality (core for PyTorch)
import torch.nn as nn  # For creating neural networks

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Get the directory of current file
dirname = os.path.dirname(__file__)


# Function to construct a dictionary of heteronyms
def construct_heteronym_dictionary():
    """
    Constructs a dictionary for heteronyms (words with multiple pronunciations based on context).
    """
    heteronym_file = os.path.join(dirname, 'g2p-assets', 'heteronyms.txt')
    heteronym2features = dict()

    with open(heteronym_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines
            headword, pron1, pron2, pos1 = line.strip().split("|")
            heteronym2features[headword.lower()] = (
                pron1.split(), pron2.split(), pos1)

    return heteronym2features


# G2P model using PyTorch
class G2P(nn.Module):
    def __init__(self):
        """
        Initializes the G2P model.
        """
        super(G2P, self).__init__()  # Required for PyTorch models

        # List of graphemes including special tokens for padding, unknown,
        # and end of sequence, respectively
        self.graphemes = ["<pad>", "<unk>", "</s>"] + \
            list("abcdefghijklmnopqrstuvwxyz")

        # List of phonemes including special tokens for padding, unknown,
        # start of sequence, and end of sequence, respectively
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
            'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
            'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',
            'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0',
            'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
            'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
            'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0',
            'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]

        # Mapping each grapheme to a unique index
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        # Mapping an index back to the corresponding grapheme
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}
        # Mapping each phoneme to a unique index
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        # Mapping an index back to the corresponding phoneme
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        # Embedding dimension for graphemes and phonemes.
        embedding_dim = 128
        # Hidden state size for GRUs (RNN units).
        hidden_size = 256
        # Dropout rate
        dropout_rate = 0.2

        # Embedding for input graphemes
        self.enc_emb = nn.Embedding(len(self.graphemes), embedding_dim)
        # GRU for encoding
        self.enc_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        # Embedding for phoneme outputs
        self.dec_emb = nn.Embedding(len(self.phonemes), embedding_dim)
        # GRU for decoding
        self.dec_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        # Fully connected layer to map hidden states to phonemes
        self.fc = nn.Linear(hidden_size, len(self.phonemes))

        # Dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)

        # Load pre-trained variables
        self.load_variables()

        # Heteronym dictionary
        self.heteronym2features = construct_heteronym_dictionary()

    def load_variables(self, checkpoint_path=None):
        """
        Loads the pre-trained model parameters.
        """
        if checkpoint_path is None:
            # Set the default checkpoint path
            checkpoint_path = os.path.join(dirname, 'g2p-assets', 'model-checkpoint.pt')
        
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
            return  # Use random weights, so no need to load anything

        # If the checkpoint exists, load it
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Load parameters into encoder layers
        self.enc_emb.weight.data.copy_(checkpoint['enc_emb.weight'])
        self.enc_gru.weight_ih_l0.data.copy_(checkpoint['enc_gru.weight_ih_l0'])
        self.enc_gru.weight_hh_l0.data.copy_(checkpoint['enc_gru.weight_hh_l0'])
        self.enc_gru.bias_ih_l0.data.copy_(checkpoint['enc_gru.bias_ih_l0'])
        self.enc_gru.bias_hh_l0.data.copy_(checkpoint['enc_gru.bias_hh_l0'])

        # Load parameters into decoder layers
        self.dec_emb.weight.data.copy_(checkpoint['dec_emb.weight'])
        self.dec_gru.weight_ih_l0.data.copy_(checkpoint['dec_gru.weight_ih_l0'])
        self.dec_gru.weight_hh_l0.data.copy_(checkpoint['dec_gru.weight_hh_l0'])
        self.dec_gru.bias_ih_l0.data.copy_(checkpoint['dec_gru.bias_ih_l0'])
        self.dec_gru.bias_hh_l0.data.copy_(checkpoint['dec_gru.bias_hh_l0'])


        # Load parameters for the fully connected layer
        self.fc.weight.data.copy_(checkpoint['fc.weight'])
        self.fc.bias.data.copy_(checkpoint['fc.bias'])

    def forward(self, x):
        """
        Forward pass for the encoder.
        """
        # Convert input graphemes into embeddings
        embedded = self.enc_emb(x)
        # Apply dropout to the embeddings
        embedded = self.dropout(embedded)
        # Passing embeddings through the GRU to get the hidden state
        _, hidden = self.enc_gru(embedded)
        return hidden

    def predict(self, word):
        """
        Predicts the phoneme sequence for a given word.
        """
        self.eval() # Set model to evaluation mode
        # Convert the word into a list of characters and end of sequence token
        chars = list(word) + ["</s>"]
        # Map characters to their corresponding indices
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        # Convert to a tensor and add batch dimension
        x = torch.tensor(x).unsqueeze(0)

        # Encoder forward pass to get hidden state
        hidden = self.forward(x)

        # Initialize decoder with the start token
        decoder_input = torch.tensor([[self.p2idx["<s>"]]])
        # List to store predicted phonemes
        preds = []
        # Max sequence length
        max_length = 50

        for _ in range(max_length):
            # Embed the current decoder input
            embedded = self.dec_emb(decoder_input)
            # Apply dropout to the decoder embedding
            embedded = self.dropout(embedded)
            # Pass through the decoder GRU
            output, hidden = self.dec_gru(embedded, hidden)
            # Pass the output through the fully connected layer
            logits = self.fc(output.squeeze(1))
            pred = logits.argmax(dim=1)  # Get the predicted phoneme
            pred_idx = pred.item()  # Convert to a Python scalar
            # If the end-of-sequence token is predicted, stop
            if pred_idx == self.p2idx["</s>"]:
                break
            preds.append(pred_idx)  # Append the predicted phoneme
            decoder_input = pred.unsqueeze(0)  # Update the decoder input

        # Convert phoneme indices back to phonemes
        preds = [self.idx2p[idx] for idx in preds]
        return preds

    def normalize_text(self, text):
        """
        Normalizes the input text to only contain lowercase a-z characters.
        """
        text = ''.join(
            # Decompose characters with diacritics
            char for char in unicodedata.normalize('NFD', text)
            # Remove diacritical marks
            if unicodedata.category(char) != 'Mn'
        )
        text = text.lower()
        text = re.sub(r"[^ a-z]", "", text)
        return text

    def __call__(self, text):
        """
        Processes input text and returns the phonetic representation.
        """
        # Normalize the input text
        text = self.normalize_text(text)
        # Use spaCy to tokenize and POS tag the text
        doc = nlp(text)
        # Extract tokens and their POS tags
        tokens = [(token.text, token.tag_) for token in doc]

        # List to store phonetic transcriptions
        prons = []
        for word, pos in tokens:
            if not re.search("[a-z]", word):
                # If the word doesn't contain letters, treat as is
                pron = [word]
            elif word.lower() in self.heteronym2features:
                pron1, pron2, pos1 = self.heteronym2features[word.lower()]
                # Choose appropriate pronunciation based on POS tag
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            else:
                pron = self.predict(word.lower())  # Predict phoneme sequence
            prons.extend(pron)  # Add pronunciation to result
            prons.append(" ")  # Separate words with space
        return prons[:-1]  # Remove the last extra space

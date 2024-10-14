import os  # For file path operations
import re  # For regular expressions
import unicodedata  # For Unicode normalization

import numpy as np  # For numerical computations
import spacy  # For tokenization and POS tagging
import torch
import torch.nn as nn

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Function to construct a dictionary of heteronyms
def construct_heteronym_dictionary():
    """
    Constructs a dictionary for heteronyms (words with multiple pronunciations based on context).
    """
    dirname = os.path.dirname(__file__)
    heteronym_file = os.path.join(dirname, 'g2p-assets', 'heteronyms.txt')
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

        # Load pre-trained variables
        self.load_variables()

        # Heteronym dictionary
        self.heteronym2features = construct_heteronym_dictionary()

    def load_variables(self):
        """
        Loads the pre-trained model parameters.
        """
        dirname = os.path.dirname(__file__)
        checkpoint = torch.load(os.path.join(dirname, 'g2p-assets', 'checkpoint.pt'), map_location='cpu', weights_only=True)
        self.enc_emb.weight.data.copy_(checkpoint['enc_emb'])
        self.enc_gru.weight_ih_l0.data.copy_(checkpoint['enc_gru_weight_ih_l0'])
        self.enc_gru.weight_hh_l0.data.copy_(checkpoint['enc_gru_weight_hh_l0'])
        self.enc_gru.bias_ih_l0.data.copy_(checkpoint['enc_gru_bias_ih_l0'])
        self.enc_gru.bias_hh_l0.data.copy_(checkpoint['enc_gru_bias_hh_l0'])

        self.dec_emb.weight.data.copy_(checkpoint['dec_emb'])
        self.dec_gru.weight_ih_l0.data.copy_(checkpoint['dec_gru_weight_ih_l0'])
        self.dec_gru.weight_hh_l0.data.copy_(checkpoint['dec_gru_weight_hh_l0'])
        self.dec_gru.bias_ih_l0.data.copy_(checkpoint['dec_gru_bias_ih_l0'])
        self.dec_gru.bias_hh_l0.data.copy_(checkpoint['dec_gru_bias_hh_l0'])

        self.fc.weight.data.copy_(checkpoint['fc_weight'])
        self.fc.bias.data.copy_(checkpoint['fc_bias'])

    def forward(self, x):
        """
        Forward pass for the encoder.
        """
        embedded = self.enc_emb(x)
        _, hidden = self.enc_gru(embedded)
        return hidden
    
    def predict(self, word):
      """
      Predicts the phoneme sequence for a given word.
      """
      # Encode the input word
      chars = list(word) + ["</s>"]
      x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
      x = torch.tensor(x).unsqueeze(0)  # Add batch dimension

      # Encoder forward pass to get hidden state
      hidden = self.forward(x)

      # Decoder initialization
      decoder_input = torch.tensor([[self.p2idx["<s>"]]])  # Start token
      preds = []
      max_length = 20

      for _ in range(max_length):
          embedded = self.dec_emb(decoder_input)
          output, hidden = self.dec_gru(embedded, hidden)  # Now `hidden` is correctly passed along
          logits = self.fc(output.squeeze(1))
          pred = logits.argmax(dim=1)
          pred_idx = pred.item()
          if pred_idx == self.p2idx["</s>"]:
              break
          preds.append(pred_idx)
          decoder_input = pred.unsqueeze(0)  # Only unsqueeze once to maintain a 3D input

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

# gibberish-generation/g2p.py
# G2P model for converting graphemes to phonemes
# Inspired by https://github.com/Kyubyong/g2p
# and https://fehiepsi.github.io/blog/grapheme-to-phoneme/

import os  # For handling file paths and checking file existence
import re  # For regular expressions, used in text normalization
import unicodedata  # For handling Unicode normalization of text

import spacy  # For tokenizing text into words using the 'en_core_web_sm' model
import torch  # Core PyTorch library for tensor operations
import torch.nn as nn  # For defining neural network layers
import torch.nn.functional as F  # For additional functions like softmax

# Load the English language model from spaCy for tokenization
nlp = spacy.load('en_core_web_sm')

# Get the directory name of the current file for relative path handling
dirname = os.path.dirname(__file__)

class G2P(nn.Module):
    """
    Grapheme-to-Phoneme (G2P) Model using an encoder-decoder architecture with attention.
    
    This model converts a sequence of graphemes (characters) into a sequence of phonemes.
    It uses embedding layers for both graphemes and phonemes, GRU layers for encoding and decoding,
    and an attention mechanism to focus on relevant parts of the input during decoding.
    """
    def __init__(self, checkpoint_path=None):
        """
        Initialize the G2P model.
        
        Args:
            checkpoint_path (str, optional): Path to a saved model checkpoint. If provided and valid,
                                             the model will load weights from this checkpoint.
        """
        super(G2P, self).__init__()

        # Define the set of graphemes (input characters) including special tokens
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")

        # Define the set of phonemes (output units) including special tokens
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
            'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
            'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',
            'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0',
            'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
            'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
            'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0',
            'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]

        # Create mappings from graphemes to indices and vice versa
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}
        
        # Create mappings from phonemes to indices and vice versa
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        # Define model hyperparameters
        embedding_dim = 500   # Dimension of embedding vectors
        hidden_size = 500     # Hidden size for GRU layers
        dropout_rate = 0.6    # Dropout rate for regularization

        # Embedding layer for graphemes (input characters)
        self.enc_emb = nn.Embedding(len(self.graphemes), embedding_dim)
        
        # Embedding layer for phonemes (output units)
        self.dec_emb = nn.Embedding(len(self.phonemes), embedding_dim)

        # GRU layer for the encoder
        self.enc_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        # GRU layer for the decoder, which takes concatenated embedding and context vector
        self.dec_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)

        # Attention mechanism: Linear layers to compute and combine attention
        self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size + embedding_dim, embedding_dim)

        # Fully connected layer to map decoder outputs to phoneme probabilities
        self.fc = nn.Linear(hidden_size, len(self.phonemes))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # Load model weights from checkpoint if provided
        self.load_variables(checkpoint_path)

    def load_variables(self, checkpoint_path):
        """
        Load model weights from a checkpoint file if the path is valid.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Checkpoint not found or not provided. Using random weights.")

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

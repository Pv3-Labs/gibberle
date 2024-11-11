# g2p.py
# G2P model for converting graphemes to phonemes

import os
import re
import unicodedata

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

nlp = spacy.load('en_core_web_sm')

dirname = os.path.dirname(__file__)

class G2P(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(G2P, self).__init__()

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

        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        embedding_dim = 500
        hidden_size = 500
        dropout_rate = 0.6

        self.enc_emb = nn.Embedding(len(self.graphemes), embedding_dim)
        self.dec_emb = nn.Embedding(len(self.phonemes), embedding_dim)

        self.enc_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        self.dec_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)

        self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size + embedding_dim, embedding_dim)

        self.fc = nn.Linear(hidden_size, len(self.phonemes))

        self.dropout = nn.Dropout(p=dropout_rate)

        self.load_variables(checkpoint_path)

    def load_variables(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Checkpoint not found or not provided. Using random weights.")

    def forward(self, x):
        embedded = self.dropout(self.enc_emb(x))
        encoder_outputs, hidden = self.enc_gru(embedded)
        return encoder_outputs, hidden

    def decode_step(self, decoder_input, decoder_hidden, encoder_outputs):
        embedded = self.dropout(self.dec_emb(decoder_input))
        attn_weights = torch.bmm(decoder_hidden.transpose(0, 1), encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=2)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, decoder_hidden = self.dec_gru(rnn_input, decoder_hidden)
        output = self.fc(output.squeeze(1))
        return output, decoder_hidden

    def predict(self, word, beam_width=3):
        self.eval()
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = torch.tensor(x).unsqueeze(0).to(next(self.parameters()).device)

        encoder_outputs, hidden = self.forward(x)
        encoder_outputs = encoder_outputs  # (1, seq_len, hidden_size)

        decoder_hidden = hidden  # (1, batch_size, hidden_size)

        decoder_input = torch.tensor([[self.p2idx["<s>"]]], device=x.device)  # (1,1)

        beam_width = beam_width
        max_length = 50

        beams = [(0, decoder_input, decoder_hidden)]
        completed_beams = []

        for _ in range(max_length):
            new_beams = []
            for score, decoder_input_seq, decoder_hidden in beams:
                last_token = decoder_input_seq[:, -1:]
                if last_token.item() == self.p2idx["</s>"]:
                    completed_beams.append((score, decoder_input_seq))
                    continue

                output, decoder_hidden_new = self.decode_step(last_token, decoder_hidden, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    new_score = score + top_log_probs[0, i].item()
                    new_decoder_input_seq = torch.cat([decoder_input_seq, top_indices[:, i:i+1]], dim=1)
                    new_beams.append((new_score, new_decoder_input_seq, decoder_hidden_new))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

            if all(decoder_input_seq[0, -1].item() == self.p2idx["</s>"] for _, decoder_input_seq, _ in beams):
                break

        completed_beams.extend(beams)
        best_beam = max(completed_beams, key=lambda x: x[0])

        output_indices = best_beam[1].squeeze().tolist()[1:]  # Exclude <s>
        if self.p2idx["</s>"] in output_indices:
            end_idx = output_indices.index(self.p2idx["</s>"])
            output_indices = output_indices[:end_idx]
        preds = [self.idx2p[idx] for idx in output_indices]
        return preds

    def normalize_text(self, text):
        text = ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )
        text = text.lower()
        text = re.sub(r"[^ a-z']", "", text)
        return text

    def __call__(self, text):
        text = self.normalize_text(text)
        doc = nlp(text)
        tokens = [token.text for token in doc]

        prons = []
        for word in tokens:
            if not re.search("[a-z]", word):
                pron = [word]
            else:
                pron = self.predict(word.lower())
            prons.extend(pron)
            prons.append(" ")
        return prons[:-1] 

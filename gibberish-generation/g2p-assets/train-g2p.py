# train-g2p.py
# For training the G2P model using the CMUdict dataset

import os
import random
import sys

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g2p import G2P

dirname = os.path.dirname(__file__)

def load_cmudict_data(cmudict_file):
    data = []
    with open(cmudict_file, 'r', encoding='latin-1') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith(';;;'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].split('(')[0]
                    phonemes = parts[1:]
                    data.append({'word': word, 'phonemes': phonemes, 'line_num': line_num})
                else:
                    print(f"Warning: Line {line_num} is invalid: {line}")
    return data

class CMUDictDataset(Dataset):
    def __init__(self, data, g2p_model):
        self.data = data
        self.g2p_model = g2p_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]['word']
        phonemes = self.data[idx]['phonemes']
        line_num = self.data[idx]['line_num']

        graphemes = list(word.lower()) + ["</s>"]
        input_ids = [self.g2p_model.g2idx.get(char, self.g2p_model.g2idx["<unk>"]) for char in graphemes]

        labels = [self.g2p_model.p2idx["<s>"]] + [self.g2p_model.p2idx.get(ph, self.g2p_model.p2idx["<unk>"]) for ph in phonemes] + [self.g2p_model.p2idx["</s>"]]

        return {'input_ids': input_ids, 'labels': labels, 'word': word, 'phonemes': phonemes, 'line_num': line_num}

def collate_batch(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=g2p_model.g2idx["<pad>"])
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=g2p_model.p2idx["<pad>"])

    for item in batch:
        if abs(len(item['input_ids']) - len(item['labels'])) > 10:
            print(f"Warning: Unusual length difference between input and label at line {item['line_num']}: {item['word']}")

    return {'input_ids': input_ids_padded, 'labels': labels_padded}

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_size = input_ids.size(0)

            encoder_outputs, encoder_hidden = model.forward(input_ids)

            decoder_input = torch.full(
                (batch_size, 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
            max_length = labels.size(1)

            all_outputs = []

            decoder_hidden = encoder_hidden

            for t in range(max_length -1):
                output, decoder_hidden = model.decode_step(decoder_input, decoder_hidden, encoder_outputs)
                all_outputs.append(output)
                decoder_input = labels[:, t+1].unsqueeze(1)

            outputs = torch.stack(all_outputs, dim=1)
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             labels[:,1:].reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    best_checkpoint_file = os.path.join(dirname, 'best-checkpoint.pt')
    last_checkpoint_file = os.path.join(dirname, 'last-checkpoint.pt')

    checkpoint_path = os.path.join(dirname, 'model-checkpoint.pt')

    g2p_model = G2P(checkpoint_path=checkpoint_path)

    device = torch.device('cuda' if cuda_available else 'cpu')
    g2p_model.to(device)

    cmudict_file = os.path.join(dirname, 'cmudict.dict')
    if not os.path.exists(cmudict_file):
        print(f"Error: CMUdict file not found at {cmudict_file}")
        sys.exit(1)
    data = load_cmudict_data(cmudict_file)

    random.shuffle(data)
    split_index = int(0.95 * len(data))
    train_data = data[:split_index]
    valid_data = data[split_index:]

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")

    train_dataset = CMUDictDataset(train_data, g2p_model)
    valid_dataset = CMUDictDataset(valid_data, g2p_model)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=collate_batch, pin_memory=True if cuda_available else False,
                              num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                              collate_fn=collate_batch, pin_memory=True if cuda_available else False,
                              num_workers=0)

    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
    optimizer = torch.optim.AdamW(g2p_model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion.to(device)

    num_epochs = 50 
    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    scaler = GradScaler() if cuda_available else None

    for epoch in range(num_epochs):
        g2p_model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_size = input_ids.size(0)

            optimizer.zero_grad()

            if cuda_available and scaler is not None:
                with autocast(device_type='cuda' if cuda_available else 'cpu'):
                    encoder_outputs, encoder_hidden = g2p_model.forward(input_ids)

                    decoder_input = labels[:, :-1]
                    decoder_targets = labels[:, 1:]

                    decoder_embedded = g2p_model.dropout(g2p_model.dec_emb(decoder_input))

                    decoder_hidden = encoder_hidden

                    outputs = []
                    decoder_hidden = decoder_hidden

                    for t in range(decoder_input.size(1)):
                        decoder_input_t = decoder_input[:, t].unsqueeze(1)
                        output, decoder_hidden = g2p_model.decode_step(decoder_input_t, decoder_hidden, encoder_outputs)
                        outputs.append(output)

                    outputs = torch.stack(outputs, dim=1)

                    loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.reshape(-1))
            else:
                encoder_outputs, encoder_hidden = g2p_model.forward(input_ids)

                decoder_input = labels[:, :-1]
                decoder_targets = labels[:, 1:]

                decoder_embedded = g2p_model.dropout(g2p_model.dec_emb(decoder_input))

                decoder_hidden = encoder_hidden

                outputs = []
                decoder_hidden = decoder_hidden

                for t in range(decoder_input.size(1)):
                    decoder_input_t = decoder_input[:, t].unsqueeze(1)
                    output, decoder_hidden = g2p_model.decode_step(decoder_input_t, decoder_hidden, encoder_outputs)
                    outputs.append(output)

                outputs = torch.stack(outputs, dim=1)

                loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.reshape(-1))

            if cuda_available and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if cuda_available and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)

            if cuda_available and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

        val_loss = evaluate(g2p_model, valid_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(g2p_model.state_dict(), best_checkpoint_file)
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    torch.save(g2p_model.state_dict(), last_checkpoint_file)

# For training g2p model
# Stores model weights in checkpoint.pt

import os
import sys

import torch
import torch.nn as nn

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset  # Used for hugging face dataset 🤗
from g2p import G2P  # g2p.py
from torch.nn.utils.rnn import \
    pad_sequence  # Used to pad sequences for batching
from torch.utils.data import DataLoader  # Provides data loading utilities
from tqdm import tqdm  # For progress bars during training and validation

# Instantiate the G2P model from g2p.py
g2p_model = G2P()

# Load the s3prl/g2p Dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 421224
#     })
# })
#
# Example: ABRIDGES AH0 B R IH1 JH AH0 Z
dataset = load_dataset("s3prl/g2p")

# Split the dataset into training and validation sets (90/10 split)
dataset = dataset['train'].train_test_split(test_size=0.1)
train_dataset = dataset['train']
valid_dataset = dataset['test']

# Preprocess function that converts the dataset's examples to input IDs and labels
def preprocess_function(examples):
    # List for storing input grapheme sequenecs
    input_ids = []
    # List to store target phoneme sequences
    labels = []
    for line in examples['text']:
        parts = line.strip().split() # Split line into word and phonemes
        if len(parts) < 2:
            continue  # Skip invalid lines
        word = parts[0].lower() # Extract word and convert to lowercase
        phonemes = parts[1:] # Extract phonemes for the word

        # Convert graphemes to indices using g2p_model mappings
        graphemes = list(word) + ["</s>"] # Add end of sequence token
        input_id = [g2p_model.g2idx
                    .get(char, g2p_model.g2idx["<unk>"])
                      for char in graphemes] # Convert graphemes to indices
        input_ids.append(input_id) # Add to input list

        # Convert phonemes to indices using g2p_model mappings
        label = [g2p_model.p2idx["<s>"]] 
        + [g2p_model.p2idx.get(ph, g2p_model.p2idx["<unk>"]) for ph in phonemes] 
        + [g2p_model.p2idx["</s>"]]
        labels.append(label) # Add to labels list

    # Return processed data as a dictionary
    return {'input_ids': input_ids, 'labels': labels}

# Apply preprocessing to both training and validation sets
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text'])

# Collate function to pad sequences to the same length for batching
def collate_batch(batch):
    # Convert input and labels to tensors
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    # Pad the input IDs and labels with <pad> tokens to ensure uniform length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=g2p_model.g2idx["<pad>"])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=g2p_model.p2idx["<pad>"])

    # Return padded sequence
    return {'input_ids': input_ids_padded, 'labels': labels_padded}

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(tokenized_datasets['test'], batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
optimizer = torch.optim.Adam(g2p_model.parameters(), lr=0.001)

# Move the model to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g2p_model.to(device)

# Evaluation function to calculate loss and accuracy on the validation set
def evaluate(model, data_loader):
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    total_loss = 0
    total_phonemes = 0
    correct_phonemes = 0
    total_words = 0
    correct_words = 0
    # Define loss function for evaluation
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"]) 

    # Disable gradient calculation for evaluation (saves memory and speeds up computations)
    with torch.no_grad():
        # Iterate over validation batches with progress bar
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device) # Move input IDs to device
            labels = batch['labels'].to(device) # Move labels to device
            batch_size = input_ids.size(0)

            # Forward pass through the encoder
            encoder_hidden = model.forward(input_ids)

             # Decoder initialization with <s> (start) token
            decoder_input = torch.full((batch_size, 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
            hidden = encoder_hidden
            # Maximum target sequence length
            max_length = labels.size(1)

            all_preds = [] # List to store predictions
            all_logits = [] # List to store logits (output probabilities)

            # Decode one phoneme at a time
            for t in range(max_length):
                decoder_embedded = model.dec_emb(decoder_input)  # Embed the decoder input
                output, hidden = model.dec_gru(decoder_embedded, hidden)  # Forward pass through GRU
                logits = model.fc(output.squeeze(1))  # Get phoneme prediction logits from the fully connected layer
                all_logits.append(logits) # Append logits
                top1 = logits.argmax(1, keepdim=True)  # Get the highest probability phoneme (argmax)
                all_preds.append(top1)  # Append prediction
                decoder_input = top1  # Use the predicted phoneme as the next input

            # Concatenate all predictions and logits
            preds = torch.cat(all_preds, dim=1)
            logits = torch.stack(all_logits, dim=1)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate phoneme-leverl accuracy
            mask = labels != g2p_model.p2idx["<pad>"] # Ignore padded tokens
            total_phonemes += mask.sum().item()
            correct_phonemes += ((preds == labels) & mask).sum().item()

            # Calculate word-level accuracy (only count if entire word is predicted correctly)
            for i in range(batch_size):
                if torch.equal(preds[i][mask[i]], labels[i][mask[i]]):
                    correct_words += 1
                total_words += 1

    # Compute average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    phoneme_accuracy = correct_phonemes / total_phonemes
    word_accuracy = correct_words / total_words
    return avg_loss, phoneme_accuracy, word_accuracy

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    g2p_model.train() # Set model to training mode
    total_loss = 0

    # Iterate over training batches
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device) # Move input IDs to device
        labels = batch['labels'].to(device) # Move labels to device

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass through the encoder
        encoder_hidden = g2p_model.forward(input_ids)

        # Inputs for the decoder (shifted by one position)
        decoder_input_ids = labels[:, :-1]
        # Target for the decoder (shifted by one position)
        decoder_target_ids = labels[:, 1:]

        # Embed the decoder inputs
        decoder_embedded = g2p_model.dec_emb(decoder_input_ids)
        # Forward pass through the decoder GRU
        output, _ = g2p_model.dec_gru(decoder_embedded, encoder_hidden)
        # Get phoneme prediction logits from the fully connected layer
        logits = g2p_model.fc(output)

        # Flatten logits and targets for computing cross-entropy loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = decoder_target_ids.reshape(-1)

        # Calculate the loss
        loss = criterion(logits_flat, targets_flat)
        loss.backward() # Backpropagate the loss 
        optimizer.step() # Update model parameters

        total_loss += loss.item() # Accumulate the total loss

    # Compute average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # Perform validation after each epoch
    val_loss, phoneme_acc, word_acc = evaluate(g2p_model, valid_loader)
    print(f"Validation Loss: {val_loss:.4f}, Phoneme Accuracy: {phoneme_acc:.4f}, Word Accuracy: {word_acc:.4f}")

# Save the trained model's parameters to a checkpoint file
checkpoint_file = os.path.join(os.path.dirname(__file__), 'checkpoint.pt')
torch.save({
    # Save encoder embedding weights
    'enc_emb': g2p_model.enc_emb.weight.data,

    # Save encoder GRU weights
    'enc_gru_weight_ih_l0': g2p_model.enc_gru.weight_ih_l0.data,
    'enc_gru_weight_hh_l0': g2p_model.enc_gru.weight_hh_l0.data,
    'enc_gru_bias_ih_l0': g2p_model.enc_gru.bias_ih_l0.data,
    'enc_gru_bias_hh_l0': g2p_model.enc_gru.bias_hh_l0.data,

    # Save decoder embedding weights
    'dec_emb': g2p_model.dec_emb.weight.data,

    # Save decoder GRU weights
    'dec_gru_weight_ih_l0': g2p_model.dec_gru.weight_ih_l0.data,
    'dec_gru_weight_hh_l0': g2p_model.dec_gru.weight_hh_l0.data,
    'dec_gru_bias_ih_l0': g2p_model.dec_gru.bias_ih_l0.data,
    'dec_gru_bias_hh_l0': g2p_model.dec_gru.bias_hh_l0.data,

    # Save fully connected layer weights
    'fc_weight': g2p_model.fc.weight.data,
    # Save fully connected layer biases
    'fc_bias': g2p_model.fc.bias.data,
}, checkpoint_file)
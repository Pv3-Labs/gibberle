# For training g2p model
# Stores model weights in checkpoint.pt

import os
import random
import sys

import torch
import torch.nn as nn

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset  # Used for hugging face dataset 🤗
from g2p import G2P  # g2p.py
from torch.amp import GradScaler, autocast  # For mixed precision
from torch.nn.utils.rnn import \
    pad_sequence  # Used to pad sequences for batching
from torch.utils.data import DataLoader  # Provides data loading utilities
from tqdm import tqdm  # For progress bars during training and validation


# Preprocess function that converts the dataset's examples to input IDs and labels
def preprocess_function(examples):
    # List for storing input grapheme sequenecs
    input_ids = []
    # List to store target phoneme sequences
    labels = []
    for line in examples['text']:
        parts = line.strip().split()  # Split line into word and phonemes
        if len(parts) < 2:
            continue  # Skip invalid lines
        word = parts[0].lower()  # Extract word and convert to lowercase
        phonemes = parts[1:]  # Extract phonemes for the word

        # Convert graphemes to indices using g2p_model mappings
        graphemes = list(word) + ["</s>"]  # Add end of sequence token
        input_id = [g2p_model.g2idx.get(
            char, g2p_model.g2idx["<unk>"]) for char in graphemes]  # Convert graphemes to indices
        input_ids.append(input_id)  # Add to input list

        # Convert phonemes to indices using g2p_model mappings
        label = [g2p_model.p2idx["<s>"]] + [g2p_model.p2idx.get(
            ph, g2p_model.p2idx["<unk>"]) for ph in phonemes] + [g2p_model.p2idx["</s>"]]
        labels.append(label)

    # Return processed data as a dictionary
    return {'input_ids': input_ids, 'labels': labels}


# Collate function to pad sequences to the same length for batching
def collate_batch(batch):
    # Convert input and labels to tensors
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    # Pad the input IDs and labels with <pad> tokens to ensure uniform length
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=g2p_model.g2idx["<pad>"])
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=g2p_model.p2idx["<pad>"])

    # Return padded sequence
    return {'input_ids': input_ids_padded, 'labels': labels_padded}


# Evaluation function to calculate loss on the validation set
def evaluate(model, data_loader):
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])

    # Disable gradient calculation for evaluation (saves memory and speeds up computations)
    with torch.no_grad():
        # Iterate over validation batches with progress bar
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(
                device)  # Move input IDs to device
            labels = batch['labels'].to(device)  # Move labels to device
            batch_size = input_ids.size(0)

            # Forward pass through the encoder
            encoder_hidden = model.forward(input_ids)

            # Decoder initialization with <s> (start) token
            decoder_input = torch.full(
                (batch_size, 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
            max_length = labels.size(1)

            all_logits = []  # List to store logits

            # Adjust hidden state if needed for evaluation (squeeze only if the batch size is 1)
            if encoder_hidden.dim() == 3 and batch_size == 1:
                encoder_hidden = encoder_hidden.squeeze(1)

            # Decode one phoneme at a time
            for t in range(max_length):
                decoder_embedded = model.dec_emb(
                    decoder_input).squeeze(1)  # Embed the decoder input

                # Forward pass through GRU (no need to unsqueeze, use 3D tensors)
                output, encoder_hidden = model.dec_gru(
                    decoder_embedded.unsqueeze(1), encoder_hidden)

                # Get phoneme prediction logits from the fully connected layer
                logits = model.fc(output.squeeze(1))
                all_logits.append(logits)

                # Greedily choose the most probable token
                decoder_input = logits.argmax(dim=-1).unsqueeze(1)

            # Concatenate all logits
            logits = torch.stack(all_logits, dim=1)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    # Compute average validation loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss


if __name__ == '__main__':
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

    # Apply preprocessing to both training and validation sets
    tokenized_datasets = dataset.map(
        preprocess_function, batched=True, remove_columns=['text'])

    # Create DataLoaders for training and validation datasets with optimizations:
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=96, shuffle=True,
                              collate_fn=collate_batch, pin_memory=True)
    valid_loader = DataLoader(tokenized_datasets['test'], batch_size=96, shuffle=False,
                              collate_fn=collate_batch, pin_memory=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
    optimizer = torch.optim.Adam(g2p_model.parameters(), lr=0.001, weight_decay=1e-4)

    # Mixed precision training setup
    scaler = GradScaler()

    # Move the model to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g2p_model.to(device)

    num_epochs = 25  # Number of epochs
    accumulation_steps = 4  # Number of batches for gradient accumulation
    best_val_loss = float('inf')  # Lowest validation loss
    patience = 10  # Number of epochs to wait for improvement
    epochs_without_improvement = 0  # Number of epochs without improvement

    # Training loop with optimizations for mixed precision and gradient accumulation
    for epoch in range(num_epochs):
        g2p_model.train()  # Set model to training mode
        total_loss = 0

        # Exponential decay for teacher forcing probability
        p_teacher_forcing = 0.9 * (0.85 ** epoch)

        # Iterate over training batches
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)  # Move input IDs to device
            labels = batch['labels'].to(device)  # Move labels to device
            batch_size = input_ids.size(0)

            # Apply noise to input_ids during training
            def add_noise_to_input(input_ids, noise_factor=0.2):
                noisy_input = input_ids.clone()
                mask = torch.rand(noisy_input.shape) < noise_factor  # Randomly choose elements to replace
                noisy_input[mask] = g2p_model.g2idx["<unk>"]  # Replace some characters with <unk>
                return noisy_input
        
            # Apply noise to input_ids
            noisy_input_ids = add_noise_to_input(input_ids)

            # Zero the gradients only at the start of each accumulation step
            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            # Forward pass through the model using mixed precision with autocast
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Forward pass through the encoder
                encoder_hidden = g2p_model.forward(noisy_input_ids)

                # Decide whether to use teacher forcing for this batch
                use_teacher_forcing = random.random() < p_teacher_forcing

                if use_teacher_forcing:
                    # Use teacher forcing: feed the target as the next input
                    decoder_inputs = labels[:, :-1]  # Exclude the last token
                    decoder_targets = labels[:, 1:]  # Exclude the first token

                    # Embed the decoder inputs
                    decoder_embedded = g2p_model.dec_emb(decoder_inputs)

                    # Run the decoder over the entire sequence
                    outputs, _ = g2p_model.dec_gru(decoder_embedded, encoder_hidden)

                    # Get phoneme prediction logits from the fully connected layer
                    logits = g2p_model.fc(outputs)

                    # Reshape logits and targets for loss calculation
                    logits_flat = logits.view(-1, logits.size(-1))
                    decoder_targets_flat = decoder_targets.reshape(-1)

                    # Calculate the loss
                    loss = criterion(logits_flat, decoder_targets_flat)
                    loss = loss / accumulation_steps  # Scale loss by the accumulation steps
                else:
                    # No teacher forcing: use the model's own predictions as the next input
                    decoder_input = torch.full((batch_size, 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
                    output_logits = []

                    for t in range(labels.size(1) - 1):
                        decoder_embedded = g2p_model.dec_emb(decoder_input)

                        output, encoder_hidden = g2p_model.dec_gru(decoder_embedded, encoder_hidden)

                        logits = g2p_model.fc(output)
                        output_logits.append(logits)

                        predicted_token = logits.argmax(dim=-1)

                        decoder_input = predicted_token

                    output_logits = torch.cat(output_logits, dim=1)
                    logits_flat = output_logits.view(-1,output_logits.size(-1))
                    decoder_target_ids = labels[:, 1:].reshape(-1)

                    loss = criterion(logits_flat, decoder_target_ids)
                    loss = loss / accumulation_steps  # Scale loss by the accumulation steps

            # Backpropagate the loss using the gradient scaler for mixed precision
            scaler.scale(loss).backward()

            # Gradient accumulation: Only update weights every `accumulation_steps` batches
            if (i + 1) % accumulation_steps == 0:
                # Apply gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)
                # Step the optimizer and update gradients
                scaler.step(optimizer)
                scaler.update()

            # Accumulate the total loss
            total_loss += loss.item()

        # Compute average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

        # Perform validation after each epoch
        val_loss = evaluate(g2p_model, valid_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset the counter if the validation loss improved
            checkpoint_file = os.path.join(os.path.dirname(__file__), 'new-checkpoint.pt')
            torch.save(g2p_model.state_dict(), checkpoint_file)
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve}/{patience}")

        # If the counter exceeds patience, stop training
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
        
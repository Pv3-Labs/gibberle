# gibberish-generation/g2p-assets/train-g2p.py
# For training the G2P model using the CMUdict dataset
# Inspired by https://fehiepsi.github.io/blog/grapheme-to-phoneme/

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

# Get the directory name of the current file for relative path handling
dirname = os.path.dirname(__file__)

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

class CMUdictDataset(Dataset):
    """
    Custom Dataset for the CMU Pronouncing Dictionary.
    
    Each item in the dataset consists of input grapheme indices and target phoneme indices.
    """
    def __init__(self, data, g2p_model):
        """
        Initialize the dataset.
        
        Args:
            data (List[dict]): The loaded CMUdict data.
            g2p_model (G2P): An instance of the G2P model to access mappings.
        """
        self.data = data
        self.g2p_model = g2p_model

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        Retrieve the sample at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing input_ids and labels.
        """
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

def evaluate(model, data_loader):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (G2P): The G2P model to evaluate.
        data_loader (DataLoader): DataLoader for the validation set.
    
    Returns:
        float: The average loss over the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass through the encoder
            encoder_outputs, encoder_hidden = model.forward(input_ids)

            # Initialize decoder input with start-of-sequence token for each sample in batch
            decoder_input = torch.full(
                (labels.size(0), 1), g2p_model.p2idx["<s>"], dtype=torch.long).to(device)
            max_length = labels.size(1)  # Maximum sequence length

            all_outputs = []

            decoder_hidden = encoder_hidden  # Initialize decoder hidden state

            # Iterate over each time step
            for t in range(max_length -1):
                # Perform a decoding step
                output, decoder_hidden = model.decode_step(decoder_input, decoder_hidden, encoder_outputs)
                all_outputs.append(output)
                # Use the true label as the next input (teacher forcing)
                decoder_input = labels[:, t+1].unsqueeze(1)

            # Stack all outputs and compute loss
            outputs = torch.stack(all_outputs, dim=1)  # Shape: (batch_size, seq_len, num_phonemes)
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             labels[:,1:].reshape(-1))
            total_loss += loss.item()

    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss

if __name__ == '__main__':
    # Check if CUDA is available for GPU acceleration
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA Available - using GPU for training.")
        device_details = torch.cuda.get_device_properties(0)
        if device_details.major >= 7:
            print("Tensor Cores Available - better performance from mixed precision for training.")
    else:
        print("CUDA Unavailable - using CPU for training.")

    # Define file paths for saving checkpoints
    best_checkpoint_file = os.path.join(dirname, 'best-checkpoint.pt')
    last_checkpoint_file = os.path.join(dirname, 'last-checkpoint.pt')

    # Path to load the initial checkpoint if available
    # Set checkpoint_path to None to use random weights
    checkpoint_path = os.path.join(dirname, 'model-checkpoint.pt')

    # Initialize the G2P model, optionally loading from a checkpoint
    g2p_model = G2P(checkpoint_path=checkpoint_path)

    # Set the device to GPU if available, else CPU
    device = torch.device('cuda' if cuda_available else 'cpu')
    g2p_model.to(device)  # Move the model to the selected device

    # Path to the CMU Pronouncing Dictionary file
    cmudict_file = os.path.join(dirname, 'cmudict.dict')

    # Load and parse the CMUdict data
    data = load_cmudict_data(cmudict_file)

    # Shuffle the data to ensure randomness
    random.shuffle(data)
    
    # Split the data into training and validation sets (95% train, 5% validation)
    split_index = int(0.95 * len(data))
    train_data = data[:split_index]
    valid_data = data[split_index:]

    # Create Dataset instances for training and validation
    train_dataset = CMUdictDataset(train_data, g2p_model)
    valid_dataset = CMUdictDataset(valid_data, g2p_model)

    # Create DataLoader instances for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=collate_batch, pin_memory=cuda_available)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                              collate_fn=collate_batch, pin_memory=cuda_available)

    # Define the loss function with padding ignored
    criterion = nn.CrossEntropyLoss(ignore_index=g2p_model.p2idx["<pad>"])
    
    # Define the optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(g2p_model.parameters(), lr=0.0005, weight_decay=1e-3)
    
    # Define a learning rate scheduler that reduces LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Move the loss criterion to the appropriate device
    criterion.to(device)

    num_epochs = 75   # Maximum number of training epochs
    best_val_loss = float('inf')  # Initialize best validation loss
    patience = 15  # Early stopping patience
    epochs_no_improve = 0  # Counter for epochs without improvement

    # Initialize GradScaler for mixed precision if using CUDA
    scaler = GradScaler() if cuda_available else None

    # Training loop over epochs
    for epoch in range(num_epochs):
        g2p_model.train()  # Set model to training mode
        total_loss = 0  # Accumulate training loss

        # Iterate over training batches with a progress bar
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)  # Move inputs to device
            labels = batch['labels'].to(device)  # Move labels to device

            optimizer.zero_grad()  # Reset gradients

            if cuda_available:
                # Use mixed precision training for efficiency
                with autocast(device_type='cuda'):
                    # Forward pass through the encoder
                    encoder_outputs, encoder_hidden = g2p_model.forward(input_ids)

                    # Prepare decoder inputs and targets for teacher forcing
                    decoder_input = labels[:, :-1]
                    decoder_targets = labels[:, 1:]

                    decoder_hidden = encoder_hidden  # Initialize decoder hidden state

                    outputs = []  # List to store decoder outputs

                    # Iterate over each time step for decoding
                    for t in range(decoder_input.size(1)):
                        decoder_input_t = decoder_input[:, t].unsqueeze(1)  # Current input
                        output, decoder_hidden = g2p_model.decode_step(decoder_input_t, decoder_hidden, encoder_outputs)
                        outputs.append(output)

                    # Stack all outputs for loss computation
                    outputs = torch.stack(outputs, dim=1)
                    
                    # Compute the loss
                    loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.reshape(-1))

                # Scale the loss for mixed precision and perform backpropagation
                scaler.scale(loss).backward()

                # Unscale gradients before clipping
                scaler.unscale_(optimizer)

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)

                # Update optimizer and scaler for mixed precision
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                # Forward pass through the encoder
                encoder_outputs, encoder_hidden = g2p_model.forward(input_ids)

                # Prepare decoder inputs and targets for teacher forcing
                decoder_input = labels[:, :-1]
                decoder_targets = labels[:, 1:]

                decoder_hidden = encoder_hidden  # Initialize decoder hidden state

                outputs = []  # List to store decoder outputs

                # Iterate over each time step for decoding
                for t in range(decoder_input.size(1)):
                    decoder_input_t = decoder_input[:, t].unsqueeze(1)  # Current input
                    output, decoder_hidden = g2p_model.decode_step(decoder_input_t, decoder_hidden, encoder_outputs)
                    outputs.append(output)

                # Stack all outputs for loss computation
                outputs = torch.stack(outputs, dim=1)
                
                # Compute the loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.reshape(-1))

                # Perform backpropagation
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(g2p_model.parameters(), max_norm=1.0)

                # Update optimizer normally
                optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Compute average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate the model on the validation set
        val_loss = evaluate(g2p_model, valid_loader)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} | Average Training Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            # If validation loss improved, save the model and reset patience counter
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(g2p_model.state_dict(), best_checkpoint_file)
        else:
            # Increment the patience counter if no improvement
            epochs_no_improve += 1
            print(f"Early Stop {epochs_no_improve}/{patience}: previous val loss greater than best val loss ({best_val_loss:.4f})")

        if epochs_no_improve >= patience:
            # Stop training early if no improvement for 'patience' epochs
            print(f"Stopping early due to no validation loss improvement after {patience} epochs.")
            break

    # Save the final model state
    torch.save(g2p_model.state_dict(), last_checkpoint_file)
    print(f"Training completed. Last checkpoint saved to {last_checkpoint_file}.")

import os
import random
import sys

import editdistance
import torch
from datasets import load_dataset
from tqdm import tqdm

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g2p import G2P

generated_seed = random.randint(0, 10000)
print(f"Generated Seed: {generated_seed}")
debug = False

# Load the G2P dataset from Hugging Face
def load_hf_dataset(subset_size=10000):
    dataset = load_dataset('s3prl/g2p')
    test_dataset = dataset['train']

    # Shuffle and take a subset of the dataset (e.g., 10k examples)
    subset = test_dataset.shuffle(seed=generated_seed).select(range(subset_size))
    
    # Prepare the data in a suitable format (graphemes, phonemes pairs)
    test_samples = [(item['text'].split()[0], " ".join(item['text'].split()[1:])) for item in subset]
    print("Dataset loaded and processed.")
    return test_samples

def clean_sequence(sequence, special_tokens):
    """
    Removes special tokens like <s>, </s>, and <pad> from a sequence.
    """
    return [token for token in sequence if token not in special_tokens]

# Function to evaluate the model with phoneme-level accuracy and edit distance
def evaluate_model(model, test_dataset):
    model.eval() 
    total_phoneme_count = 0
    correct_phoneme_count = 0
    total_edit_distance = 0
    total_sequences = len(test_dataset)

    # Define special tokens to be ignored during accuracy calculation
    special_tokens = ["<pad>", "<s>", "</s>"]

    with torch.no_grad():
        for i, (graphemes, target_phonemes) in enumerate(tqdm(test_dataset, desc="Processing samples")):
            predicted_phonemes = model.predict(graphemes)  # Predict using the model's predict method

            # Split target phonemes into a list for comparison
            target_phonemes_list = target_phonemes.split()

            # Clean both predicted and target sequences by removing special tokens
            predicted_phonemes_cleaned = clean_sequence(predicted_phonemes, special_tokens)
            target_phonemes_cleaned = clean_sequence(target_phonemes_list, special_tokens)

            if debug and i < 5:  # Print only for the first 5 examples
                print(f"\nExample {i+1}:")
                print(f"Graphemes: {graphemes}")
                print(f"Predicted: {predicted_phonemes_cleaned}")
                print(f"Target: {target_phonemes_cleaned}\n")
            
            # Phoneme-level accuracy
            correct_phoneme_count += sum(1 for pred, target in zip(predicted_phonemes_cleaned, target_phonemes_cleaned) if pred == target)
            total_phoneme_count += len(target_phonemes_cleaned)

            # Edit distance
            total_edit_distance += editdistance.eval(predicted_phonemes_cleaned, target_phonemes_cleaned)

    # Phoneme-level accuracy (how many phonemes were correct)
    phoneme_level_accuracy = correct_phoneme_count / total_phoneme_count if total_phoneme_count > 0 else 0

    # Average edit distance per sequence
    average_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0

    return phoneme_level_accuracy, average_edit_distance

# Function to load the G2P model and load the weights from the checkpoint
def load_g2p_model(checkpoint_path):
    model = G2P()  # Instantiate the G2P model
    model.load_variables(checkpoint_path=checkpoint_path)  # Load weights from the checkpoint
    print(f"Weights loaded from {checkpoint_path}.")
    return model


# Compare the performance of two checkpoints
def compare_checkpoints(checkpoint1_path, checkpoint2_path, subset_size=10000):
    # Load both models with their respective checkpoints
    model1 = load_g2p_model(checkpoint1_path)
    model2 = load_g2p_model(checkpoint2_path)

    # Load a subset of the dataset
    test_dataset = load_hf_dataset(subset_size)

    # Evaluate both models
    print(f"Evaluating current weights from {checkpoint1_path}:")
    accuracy1, edit_distance1 = evaluate_model(model1, test_dataset)

    print(f"Evaluating new weights from {checkpoint2_path}:")
    accuracy2, edit_distance2 = evaluate_model(model2, test_dataset)

    # Print results and comparison
    print(f"Current weights - Phoneme-level Accuracy: {accuracy1 * 100:.2f}%, Average Edit Distance: {edit_distance1:.2f}")
    print(f"New weights - Phoneme-level Accuracy: {accuracy2 * 100:.2f}%, Average Edit Distance: {edit_distance2:.2f}")

    if accuracy1 > accuracy2:
        print("Current weights perform better.")
    elif accuracy2 > accuracy1:
        print("New weights perform better.")
    else:
        print("Both weights have the same accuracy.")

    if edit_distance1 < edit_distance2:
        print("Current weights have a smaller edit distance (better performance).")
    elif edit_distance2 < edit_distance1:
        print("New weights have a smaller edit distance (better performance).")
    else:
        print("Both weights have the same average edit distance.")

# Main function to call comparison
if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    model_checkpoint = os.path.join(dirname, 'model-checkpoint.pt')
    new_checkpoint = os.path.join(dirname, 'new-checkpoint.pt')

    # Compare both checkpoints using a subset of 10,000 samples
    compare_checkpoints(model_checkpoint, new_checkpoint, subset_size=10000)

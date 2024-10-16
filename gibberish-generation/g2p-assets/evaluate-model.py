# evaluate-model.py
# For evaluation of model weights

import os
import random
import sys

import editdistance
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g2p import G2P


def load_hf_dataset(subset_size=10000, seed=0):
    dataset = load_dataset('s3prl/g2p')
    test_dataset = dataset['train']

    subset = test_dataset.shuffle(seed=seed).select(range(subset_size))
    test_samples = [(item['text'].split()[0], " ".join(item['text'].split()[1:])) for item in subset]

    print("Dataset loaded and processed.")
    return test_samples

def clean_sequence(sequence, special_tokens):
    return [token for token in sequence if token not in special_tokens]

def evaluate_model(checkpoint_path):
    model = G2P()  
    model.load_variables(checkpoint_path=checkpoint_path) 
    print(f"Weights loaded from {checkpoint_path}")
    model.eval() 
    total_phoneme_count = 0
    correct_phoneme_count = 0
    total_edit_distance = 0
    total_sequences = len(test_dataset)

    special_tokens = ["<pad>", "<s>", "</s>"]

    with torch.no_grad():
        for i, (graphemes, target_phonemes) in enumerate(tqdm(test_dataset, desc="Processing samples")):
            predicted_phonemes = model.predict(graphemes)  # Predict using the model's predict method

            target_phonemes_list = target_phonemes.split()

            predicted_phonemes_cleaned = clean_sequence(predicted_phonemes, special_tokens)
            target_phonemes_cleaned = clean_sequence(target_phonemes_list, special_tokens)

            if debug and i < 5: 
                print(f"\nExample {i+1}:")
                print(f"Graphemes: {graphemes}")
                print(f"Predicted: {predicted_phonemes_cleaned}")
                print(f"Target: {target_phonemes_cleaned}\n")
            
            correct_phoneme_count += sum(1 for pred, target in zip(predicted_phonemes_cleaned, target_phonemes_cleaned) if pred == target)
            total_phoneme_count += len(target_phonemes_cleaned)

            total_edit_distance += editdistance.eval(predicted_phonemes_cleaned, target_phonemes_cleaned)

    phoneme_level_accuracy = correct_phoneme_count / total_phoneme_count if total_phoneme_count > 0 else 0
    average_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0

    return phoneme_level_accuracy, average_edit_distance

if __name__ == "__main__":
    generated_seed = random.randint(0, 10000)
    print(f"Generated Seed: {generated_seed}")
    
    subset_size = 10000
    test_dataset = load_hf_dataset(subset_size, generated_seed)

    dirname = os.path.dirname(__file__)
    model_checkpoint = os.path.join(dirname, 'model-checkpoint.pt')
    new_checkpoint = os.path.join(dirname, 'new-checkpoint.pt')

    debug = False  # Set to true for five examples to be printed
    comparison = False  # Set to true to compare model-checkpoint and new-checkpoint weights

    if comparison:
        accuracy1, edit_distance1 = evaluate_model(model_checkpoint)
        accuracy2, edit_distance2 = evaluate_model(new_checkpoint)
        print(f"\nmodel-checkpoint.pt | Phoneme-level Accuracy: {accuracy1 * 100:.2f}%, Average Edit Distance: {edit_distance1:.2f}")
        print(f"new-checkpoint.pt | Phoneme-level Accuracy: {accuracy2 * 100:.2f}%, Average Edit Distance: {edit_distance2:.2f}\n")
    else:
        accuracy, edit_distance = evaluate_model(model_checkpoint)
        print(f"\nmodel-checkpoint.pt | Phoneme-level Accuracy: {accuracy * 100:.2f}%, Average Edit Distance: {edit_distance:.2f}\n")
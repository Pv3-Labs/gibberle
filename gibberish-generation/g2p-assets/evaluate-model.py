# gibberish-generation/g2p-assets/evaluate-model.py
# For evaluation of model weights using the CMUdict dataset

import os
import random
import sys

import editdistance
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g2p import G2P


def load_cmudict_data(cmudict_file, subset_size=10000, seed=0):
    data = []
    with open(cmudict_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(';;;'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].split('(')[0]
                    phonemes = parts[1:]
                    data.append((word.lower(), phonemes))
    random.seed(seed)
    random.shuffle(data)
    subset = data[:subset_size]
    return subset

def evaluate_model(checkpoint_path):
    model = G2P(checkpoint_path=checkpoint_path)
    model.eval()
    total_phoneme_count = 0
    correct_phoneme_count = 0
    total_edit_distance = 0
    total_sequences = len(test_dataset)
    correct_word_count = 0 

    special_tokens = ["<pad>", "<s>", "</s>"]

    with torch.no_grad():
        for i, (word, target_phonemes_list) in enumerate(tqdm(test_dataset, desc="Processing samples")):
            predicted_phonemes = model.predict(word)
            predicted_phonemes_cleaned = [ph for ph in predicted_phonemes if ph not in special_tokens]
            target_phonemes_cleaned = [ph for ph in target_phonemes_list if ph not in special_tokens]

            if debug and i < 5:
                print(f"\nExample {i+1}:")
                print(f"Word: {word}")
                print(f"Predicted: {predicted_phonemes_cleaned}")
                print(f"Target: {target_phonemes_cleaned}\n")

            correct_phoneme_count += sum(1 for pred, target in zip(predicted_phonemes_cleaned, target_phonemes_cleaned) if pred == target)
            total_phoneme_count += len(target_phonemes_cleaned)
            total_edit_distance += editdistance.eval(predicted_phonemes_cleaned, target_phonemes_cleaned)

            if predicted_phonemes_cleaned == target_phonemes_cleaned:
                correct_word_count += 1

    phoneme_level_accuracy = correct_phoneme_count / total_phoneme_count if total_phoneme_count > 0 else 0
    average_edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0
    word_level_accuracy = correct_word_count / total_sequences if total_sequences > 0 else 0

    return phoneme_level_accuracy, average_edit_distance, word_level_accuracy

if __name__ == "__main__":
    generated_seed = random.randint(0, 10000)
    print(f"Generated Seed: {generated_seed}")

    subset_size = 10000
    dirname = os.path.dirname(__file__)
    cmudict_file = os.path.join(dirname, 'cmudict.dict')
    test_dataset = load_cmudict_data(cmudict_file, subset_size, generated_seed)

    model_checkpoint = os.path.join(dirname, 'model-checkpoint.pt')
    best_checkpoint = os.path.join(dirname, 'best-checkpoint.pt')
    last_checkpoint = os.path.join(dirname, 'last-checkpoint.pt')

    debug = False  # Set to true for five examples to be printed
    comparison = False  # Set to true to compare model-checkpoint and new-checkpoint weights

    if comparison:
        accuracy1, edit_distance1, word_acc1 = evaluate_model(model_checkpoint)
        accuracy2, edit_distance2, word_acc2 = evaluate_model(best_checkpoint)
        accuracy3, edit_distance3, word_acc3 = evaluate_model(last_checkpoint)
        print(f"\nmodel-checkpoint.pt | Phoneme-level Accuracy: {accuracy1 * 100:.2f}%, Average Edit Distance: {edit_distance1:.2f}, Word-level Accuracy: {word_acc1 * 100:.2f}%")
        print(f"best-checkpoint.pt | Phoneme-level Accuracy: {accuracy2 * 100:.2f}%, Average Edit Distance: {edit_distance2:.2f}, Word-level Accuracy: {word_acc2 * 100:.2f}%")
        print(f"last-checkpoint.pt | Phoneme-level Accuracy: {accuracy3 * 100:.2f}%, Average Edit Distance: {edit_distance3:.2f}, Word-level Accuracy: {word_acc3 * 100:.2f}%")
    else:
        accuracy, edit_distance, word_acc = evaluate_model(model_checkpoint)
        print(f"\nmodel-checkpoint.pt | Phoneme-level Accuracy: {accuracy * 100:.2f}%, Average Edit Distance: {edit_distance:.2f}, Word-level Accuracy: {word_acc * 100:.2f}%\n")

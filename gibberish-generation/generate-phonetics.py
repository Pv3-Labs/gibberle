# gibberish-generation/generate-phonetics.py
# For utilizing the G2P model and comparing to a direct lookup

import os  # For handling file paths

from g2p import G2P  # Import the custom G2P model for phonetic conversion


def main():
    """
    Main function to load the G2P model, generate phonetic transcriptions for a given text,
    and compare the results with the CMU Pronouncing Dictionary.
    """
    # Get the directory of the current file for relative path handling
    dirname = os.path.dirname(__file__)
    
    # Define the path to the G2P model checkpoint
    model_checkpoint = os.path.join(dirname, 'g2p-assets', 'model-checkpoint.pt')
    
    # Load the G2P model with the checkpoint
    model = G2P(checkpoint_path=model_checkpoint)
    
    # Define the input text to be converted into phonetics
    text = "Thanks for reading"

    # Use the G2P model to generate the phonetic sequence for the input text
    phonetic_sequence = model(text)
    
    # Display original input
    print("\nOriginal Text:")
    print(text)

    # Convert the phonetic sequence list to a single string
    phonetic_output_g2p = ' '.join(phonetic_sequence)
    print("\nPhonetic representation:")
    print(phonetic_output_g2p + "\n")  # Display the G2P model's phonetic transcription

if __name__ == "__main__":
    main() 

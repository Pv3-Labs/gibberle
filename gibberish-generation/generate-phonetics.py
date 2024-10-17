# gibberish-generation/generate-phonetics.py
# For utilizing the G2P model and comparing to a direct lookup

import os  # For handling file paths

import nltk  # Natural Language Toolkit, used for accessing the CMU Pronouncing Dictionary
from g2p import G2P  # Import the custom G2P model for phonetic conversion
from nltk.corpus import \
    cmudict  # Import the CMU Pronouncing Dictionary from NLTK

# Download the CMU Pronouncing Dictionary if not already available
nltk.download('cmudict')

# Load the CMU Pronouncing Dictionary, which maps words to their phonetic representations
cmu_dict = cmudict.dict()

def lookup_cmudict(word):
    """
    Look up the phonetic transcription of a word using the CMU Pronouncing Dictionary.
    
    Args:
        word (str): The word to look up in the CMUdict.
    
    Returns:
        list: The phonetic transcription of the word as a list of phonemes,
              or None if the word is not found.
    """
    word = word.lower() 
    if word in cmu_dict:
        return cmu_dict[word][0]  # Return the first pronunciation if multiple exist
    else:
        return None  # Return None if the word is not found in CMUdict

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
    text = "Please do not touch"

    # Use the G2P model to generate the phonetic sequence for the input text
    phonetic_sequence = model(text)
    
    # Convert the phonetic sequence list to a single string
    phonetic_output_g2p = ' '.join(phonetic_sequence)
    print("\nPhonetic representation (G2P):")
    print(phonetic_output_g2p)  # Display the G2P model's phonetic transcription

    # Split the input text into individual words for CMUdict lookup
    words = text.split()
    phonetic_output_cmu = []  # List to store CMUdict phonetic transcriptions

    # Loop through each word and retrieve its phonetic transcription from CMUdict
    for word in words:
        cmu_pron = lookup_cmudict(word)  # Look up phonetic transcription in CMUdict
        if cmu_pron:
            phonetic_output_cmu.append(' '.join(cmu_pron))  # Add transcription to the list
        else:
            phonetic_output_cmu.append("[Not found in CMUdict]")  # Indicate if not found

    # Combine the phonetic transcriptions of each word with a space between them
    phonetic_output_cmu = '   '.join(phonetic_output_cmu)
    print("\nPhonetic representation (CMUdict):")
    print(phonetic_output_cmu)  # Display the CMUdict phonetic transcription

if __name__ == "__main__":
    main() 

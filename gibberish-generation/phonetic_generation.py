import json  # For writing to a JSON file
import os  # For handling file paths

from g2p import G2P  # Import the G2P model for phonetic conversion


def main():
    """
    Main function to load the G2P model, generate phonetic transcriptions for a list of phrases,
    and save the results with hints to a JSON file.
    """
    # Get the directory of the current file for relative path handling
    dirname = os.path.dirname(__file__)
    
    # Define the path to the G2P model checkpoint
    model_checkpoint = os.path.join(dirname, 'g2p-assets', 'model-checkpoint.pt')
    
    # Load the G2P model with the checkpoint
    model = G2P(checkpoint_path=model_checkpoint)
    
    # Define the input list of phrases with hints
    phrases = [
        {"phrase": "To be, or not to be", "hint": "Shakespeare"},
        {"phrase": "May the Force be with you.", "hint": "Star Wars"},
    ]
    
    # List to store the output data
    output_data = []

    # Process each phrase in the list
    for item in phrases:
        phrase_text = item["phrase"]
        hint = item["hint"]

        # Use the G2P model to generate the phonetic sequence for the input text
        phonetic_sequence = model(phrase_text)
        phonetic_output_g2p = ' '.join(phonetic_sequence)

        # Append the result to the output data list
        output_data.append({
            "phrase": phrase_text,
            "phonetic_output": phonetic_output_g2p,
            "hint": hint
        })

    # Define the output JSON file path
    output_file_path = os.path.join(dirname, 'phonetic_output.json')

    # Write the output data to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    print(f"Phonetic representations have been saved to {output_file_path}")

if __name__ == "__main__":
    main()

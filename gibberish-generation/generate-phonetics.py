import nltk
from g2p import G2P
# from g2p_en import G2P
from nltk.corpus import cmudict

# Download cmudict if not already installed
nltk.download('cmudict')

# Load the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Function to lookup word pronunciation in CMUdict
def lookup_cmudict(word):
    word = word.lower()
    if word in cmu_dict:
        return cmu_dict[word][0]  # Taking the first pronunciation if multiple exist
    else:
        return None

def main():
    # Initialize G2P model
    model = G2P()

    # Input text
    text = "Please do not touch"
    
    # Get G2P model's phonetic representation
    phonetic_sequence = model(text)
    phonetic_output_g2p = ' '.join(phonetic_sequence)
    print("\nPhonetic representation (G2P):")
    print(phonetic_output_g2p)

    # Split text into words
    words = text.split()

    # Get CMUdict phonetic representation
    phonetic_output_cmu = []
    for word in words:
        cmu_pron = lookup_cmudict(word)
        if cmu_pron:
            phonetic_output_cmu.append(' '.join(cmu_pron))
        else:
            phonetic_output_cmu.append("[Not found in CMUdict]")
    
    phonetic_output_cmu = '   '.join(phonetic_output_cmu)
    print("\nPhonetic representation (CMUdict):")
    print(phonetic_output_cmu)

if __name__ == "__main__":
    main()

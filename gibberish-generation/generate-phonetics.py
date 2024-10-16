# generate-phonetics.py
# For utilizing the G2P model

import nltk
from g2p import G2P
from nltk.corpus import cmudict

nltk.download('cmudict')
cmu_dict = cmudict.dict()


def lookup_cmudict(word):
    word = word.lower()
    if word in cmu_dict:
        return cmu_dict[word][0]
    else:
        return None

def main():
    model = G2P()
    text = "Please do not touch"

    phonetic_sequence = model(text)
    phonetic_output_g2p = ' '.join(phonetic_sequence)
    print("\nPhonetic representation (G2P):")
    print(phonetic_output_g2p)

    words = text.split()
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
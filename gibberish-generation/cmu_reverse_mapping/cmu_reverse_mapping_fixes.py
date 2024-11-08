import tensorflow as tf
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, regexp_replace, length, trim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dot, Activation, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping




# Initialize SparkSession
spark = SparkSession.builder.appName("PhonemeToGrapheme").getOrCreate()

# Read the file as a DataFrame with a single column 'value'
df = spark.read.text('cmudict-0.7b-utf8.txt')

# Filter out comment lines
df = df.filter(~col('value').startswith(';;;'))

# Define a regular expression pattern to match the word and phonemes
pattern = r'^(\S+)\s+(.*)$'

# Use regexp_extract to extract 'word' and 'phonemes'
cmu_df = df.select(
    regexp_extract('value', pattern, 1).alias('word'),
    regexp_extract('value', pattern, 2).alias('phonemes')
)

# Remove non-alphanumeric characters from 'word'
cmu_df = cmu_df.withColumn('word', regexp_replace('word', '[^a-zA-Z0-9]', ''))

# Filter out rows with null or empty 'word' or 'phonemes'
cmu_df = cmu_df.filter(
    (cmu_df.word.isNotNull()) &
    (cmu_df.phonemes.isNotNull()) &
    (length(trim(col('word'))) > 0) &
    (length(trim(col('phonemes'))) > 0)
)

# Step 1: Filter out words containing digits or symbols
pattern = r'^[a-zA-Z]+$'
cmu_df = cmu_df.filter(col('word').rlike(pattern))

# Step 2: Remove any remaining non-alphabetic characters from 'word'
cmu_df = cmu_df.withColumn('word', regexp_replace('word', '[^a-zA-Z]', ''))

# Step 3: Remove rows where 'word' is empty or null
cmu_df = cmu_df.filter((col('word').isNotNull()) & (length(trim(col('word'))) > 0))

# Split data into 70% train, 15% validation, and 15% test
train_df, test_df = cmu_df.randomSplit([0.8, 0.2], seed=42)

train_df = train_df.limit(105600)

test_df = test_df.limit(26400)

# Import necessary libraries

# Define tokens to be added
start_token = "<start>"
end_token = "<end>"

# Tokenize phonemes normally, and graphemes at the character level with start and end tokens included
train_phonemes = [row['phonemes'] for row in train_df.collect()]
train_graphemes = [f"{start_token} {' '.join(row['word'])} {end_token}" for row in train_df.collect()]

# Initialize and fit tokenizers
phoneme_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
phoneme_tokenizer.fit_on_texts(train_phonemes)

# Character-level tokenizer for graphemes
grapheme_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True, lower=False)
grapheme_tokenizer.fit_on_texts(train_graphemes)

# Get vocabulary sizes
phoneme_vocab_size = len(phoneme_tokenizer.word_index) + 1  # +1 for padding token
grapheme_vocab_size = len(grapheme_tokenizer.word_index) + 1

# Convert to numerical sequences
train_phoneme_seq = phoneme_tokenizer.texts_to_sequences(train_phonemes)
train_grapheme_seq = grapheme_tokenizer.texts_to_sequences(train_graphemes)

# Step 1: Calculate the correct sequence lengths
encoder_seq_length = max(len(seq) for seq in train_phoneme_seq)
decoder_seq_length = max(len(seq) for seq in train_grapheme_seq)  # No -1 here; we pad to the full length

# Set the correct sequence length based on the model's expectations
decoder_seq_length = 27  # Adjust based on your model's requirement or maximum decoder sequence length

# Pad the sequences to ensure consistent shapes
X_train_padded = pad_sequences(train_phoneme_seq, maxlen=encoder_seq_length, padding='post')
Y_train_padded = pad_sequences(train_grapheme_seq, maxlen=decoder_seq_length, padding='post')

# Align decoder input and target data shapes
decoder_input_data = Y_train_padded[:, :-1]  # For input, remove the last token
decoder_target_data = Y_train_padded[:, 1:]  # For target, remove the first token

# Step 3: Align decoder input and target sequences
decoder_input_data = Y_train_padded[:, :-1]  # Remove last token for decoder input
decoder_target_data = Y_train_padded[:, 1:]  # Remove first token for decoder target

# Step 4: Verify tokenization and padding
# Print vocabulary sizes
print("Phoneme vocabulary size:", phoneme_vocab_size)
print("Grapheme vocabulary size:", grapheme_vocab_size)

# Print encoder and decoder shapes
print("Encoder input shape:", X_train_padded.shape)  # (num_samples, encoder_seq_length)
print("Decoder input shape:", decoder_input_data.shape)  # (num_samples, decoder_seq_length - 1)
print("Decoder target shape:", decoder_target_data.shape)  # (num_samples, decoder_seq_length - 1)

# Verify a few tokenized sequences to ensure they look correct
for i, seq in enumerate(train_grapheme_seq[:5]):
    print(f"Original grapheme sequence {i+1}: {train_graphemes[i]}")
    print(f"Tokenized grapheme sequence {i+1}: {seq}")


# Define vocabulary sizes for readability
encoder_vocab_size = len(phoneme_tokenizer.word_index) + 1
decoder_vocab_size = len(grapheme_tokenizer.word_index) + 1
embedding_dim = 128
lstm_units = 512

# Encoder model definition
encoder_input = Input(shape=(sequence_length,), name="encoder_input")
encoder_embedding = Embedding(input_dim=encoder_vocab_size, output_dim=embedding_dim)(encoder_input)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(encoder_embedding)
encoder_model = Model(encoder_input, [encoder_outputs, state_h, state_c])

# Decoder model setup
decoder_input = Input(shape=(sequence_length - 1,), name="decoder_input")
decoder_embedding = Embedding(input_dim=decoder_vocab_size, output_dim=embedding_dim)(decoder_input)
decoder_lstm, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(decoder_embedding, initial_state=[state_h, state_c])

# Attention mechanism
# Calculate attention scores and weights
attention_scores = Dot(axes=[2, 2])([decoder_lstm, encoder_outputs])  # Shape (batch, decoder_seq_length, encoder_seq_length)
attention_weights = Activation("softmax")(attention_scores)  # Shape (batch, decoder_seq_length, encoder_seq_length)
context_vector = Dot(axes=[2, 1])([attention_weights, encoder_outputs])  # Shape (batch, decoder_seq_length, lstm_units)

# Concatenate context vector and decoder output
decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_lstm])

# Output layer with TimeDistributed
output = TimeDistributed(Dense(decoder_vocab_size, activation="softmax"))(decoder_combined_context)

# Define and compile the full model
model = Model([encoder_input, decoder_input], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with aligned input and target shapes
history = model.fit(
    {'encoder_input': X_train_padded, 'decoder_input': decoder_input_data},
    decoder_target_data,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping]
)

def predict_word(phoneme_sequence, encoder_model, decoder_model, phoneme_tokenizer, grapheme_tokenizer, max_len_input, max_len_output, beam_width=3):
    # Preprocess the input phoneme sequence
    input_seq = phoneme_tokenizer.texts_to_sequences([phoneme_sequence])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_len_input, padding='post')

    # Encode the input sequence to get the encoder outputs and states
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq_padded)
    
    # Initialize the decoder input with the start token
    start_token = grapheme_tokenizer.word_index['\t']
    end_token = grapheme_tokenizer.word_index['\n']
    
    # Initialize beam search with initial states
    beams = [(state_h, state_c, [start_token], 1.0)]  # (state_h, state_c, sequence, probability)

    for _ in range(max_len_output):
        new_beams = []
        for state_h, state_c, seq, prob in beams:
            target_seq = np.array([seq[-1]]).reshape(1, 1)
            # Pass encoder outputs and decoder states correctly
            decoder_output, h, c = decoder_model.predict(
                [target_seq, encoder_outputs, state_h, state_c]
            )
            
            # Select top-k probabilities for beam search
            top_k_indices = np.argsort(decoder_output[0, -1, :])[-beam_width:]
            for index in top_k_indices:
                new_seq = seq + [index]
                new_prob = prob * decoder_output[0, -1, index]
                new_beams.append((h, c, new_seq, new_prob))
        
        # Keep only the top-k beams
        beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_width]

    # Choose the best beam with the highest probability
    best_seq = beams[0][2]
    # Filter out the start and end tokens from the sequence
    predicted_chars = [grapheme_tokenizer.index_word[i] for i in best_seq if i not in [start_token, end_token]]
    return ''.join(predicted_chars)


# Sample input for testing (new phoneme sequence)
sample_phoneme_sequence = 'H EH L OW'  # Example phoneme sequence for "HELLO"

# Predict the word
predicted_word = predict_word(
    sample_phoneme_sequence,
    encoder_model,
    decoder_model,
    phoneme_tokenizer,
    grapheme_tokenizer,
    max_len_input,
    max_len_output
)

print(f"Predicted word: {predicted_word}")



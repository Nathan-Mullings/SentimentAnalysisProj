import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cleantext

# Load data
data = pd.read_csv('train_cleaned.csv')
data['text'] = data['text'].fillna('')  # Handle NaN values
data['text'] = data['text'].astype(str)  # Ensure all entries are strings

# Parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

# Extracting data into lists
sentences = data['text'].tolist()
labels = data['sentiment_score'].tolist()

# Splitting the data
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type, truncating=trunc_type)

# Converting to numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels).astype(np.int32)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels).astype(np.int32)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')    # Output layer for three categories
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=2)

# Function to convert predictions to labels
def print_predictions(predictions):
    class_labels = ['negative', 'neutral', 'positive']
    return [class_labels[np.argmax(p)] for p in predictions][0]

# Streamlit application for user interaction and sentiment analysis
st.header('Sentiment Analysis Program')
# Info button and disclaimer
if st.button('Show Disclaimer'):
    st.info("Disclaimer: This sentiment analysis tool is for educational and research purposes only."
            " The predictions it makes are based on the data it has been trained on and may not accurately reflect all nuances of human emotions expressed in text.")

with st.expander('Analyse Text'):
    user_input = st.text_input('Enter your text')
    if user_input:
        # Use cleantext with supported parameters
        clean_text = cleantext.clean(user_input, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True, stp_lang='english')
        sequences = tokenizer.texts_to_sequences([clean_text])
        if sequences[0]:  # Ensure the sequence is not empty
            padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            predictions = model.predict(padded)
            sentiment = print_predictions(predictions)
            st.write("Overall Sentiment:", sentiment)
        else:
            st.write("No valid input detected. Please check your text.")

with st.expander('Analyse CSV'):
    uploaded_file = st.file_uploader('Upload your text file in CSV format', type='csv')
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error('Uploaded CSV file does not contain a "text" column.')
            else:
                df['sentiment'] = df['text'].apply(lambda x: print_predictions(model.predict(pad_sequences(tokenizer.texts_to_sequences([x]), maxlen=max_length, padding=padding_type, truncating=trunc_type))))
                st.write(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download data as CSV", data=csv, file_name='sentiment_analysis_results.csv', mime='text/csv')
        except Exception as e:
            st.error(f"An error occurred when processing the CSV file: {str(e)}")
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import brown
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Bidirectional
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.models import load_model
# from static.nlp_model_source_code import word_tokenizer
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Bidirectional
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy



nltk.download('brown')
nltk.download('universal_tagset')

corpus = brown.tagged_sents(tagset='universal')

inputs = []
targets = []

for sentence_tag_pairs in corpus:
  tokens = []
  target = []
  # print(sentence_tag_pairs)
  for token, tag in sentence_tag_pairs:
    tokens.append(token)
    target.append(tag)
  inputs.append(tokens)
  targets.append(target)

train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.3)

# Convert sentences to sequences

MAX_VOCAB_SIZE = None

should_lowercase = False
word_tokenizer = Tokenizer(
  num_words=MAX_VOCAB_SIZE,
  lower=should_lowercase,
  oov_token='UNK',
)
# otherwise unknown tokens will be removed and len(input) != len(target)
# input words and target words will not be aligned!

word_tokenizer.fit_on_texts(train_inputs)
model = load_model('static/pos_tag_model.h5')

# MAX_VOCAB_SIZE = None
# should_lowercase = False
# word_tokenizer = Tokenizer(
#     num_words=MAX_VOCAB_SIZE,
#     lower=should_lowercase,
#     oov_token='UNK',
# )

target_dict = {1:"NOUN", 2:"VERB", 3:".", 4:"ADPOSITION", 5:"DETERMINER", 6:"ADJECTIVE", 7:"ADVERB", 8:"PRONOUN", 9:"CONJUNCTION", 10:"PRT", 11:"NUM", 12:"X"}



app = Flask(__name__)

app.secret_key = "secret key"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///word_dict.db"
db = SQLAlchemy(app)

class word_db(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(200), nullable=False)
    tag = db.Column(db.String(200), nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.word}"


# db.create_all()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        word = request.form['word']
        word = [[word]]
        # word_tokenizer.fit_on_texts(word)
        train_inputs_int = word_tokenizer.texts_to_sequences(word)
        T = 180
        train_inputs_int = pad_sequences(train_inputs_int, maxlen=T)

        # first get length of each sequence
        train_lengths = []
        for sentence in word:
            train_lengths.append(len(sentence))

        train_probs = model.predict(train_inputs_int)  # N x T x K
        train_predictions = []
        for probs, length in zip(train_probs, train_lengths):
            # probs is T x K
            probs_ = probs[-length:]
            preds = np.argmax(probs_, axis=1)
            train_predictions.append(preds)

        word = np.asarray(word)
        word = np.squeeze(word).tolist()
        print(word, train_predictions[0][0])
        word_table_row = word_db(word=word, tag=f"{target_dict[train_predictions[0][0]]}")
        db.session.add(word_table_row)
        db.session.commit()
    word_table_total = word_db.query.all()
    return render_template('index.html', word_table_total=word_table_total)

@app.route('/clear', methods=['GET', 'POST'])
def clear_data():
    meta = db.metadata
    for table in reversed(meta.sorted_tables):
        # print'Clear table %s' % table
        db.session.execute(table.delete())
    db.session.commit()
    word_table_total = word_db.query.all()
    return render_template('index.html', word_table_total=word_table_total)

if __name__ == "__main__":
    # ngrok_auth_token = '2Rk3AjGV9pr55DzpSqj2DpsEPQp_3UrmnK9AJ2rsLwiAv66sV'
    # os.system(f'ngrok authtoken {ngrok_auth_token}')
    app.run()
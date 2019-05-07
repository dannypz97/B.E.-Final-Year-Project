import re
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import  pickle

WORD_INDEX = None
LABEL_ENCODER = None
LABEL_COUNT   = None
def clean_str(string):
    """
    Cleans each string and convert to lower case.
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "n not", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r"\\n", "", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def loader_encoder(table, type="json"):
    """
    Load and encode data from dataset.

    type = "sql" means get data from MySQL database.
    type = "json" means get data from .json file.        """


    if type == "json":
        with open('./data/' + table + '.json', 'r', encoding='utf8') as f:
            datastore = json.load(f)
            questions = []
            tags = []
            iter = 0

            for row in datastore:
                questions.append(clean_str(row['question']))
                tags.append(row['tags'].split(','))
                iter = iter + 1

    global WORD_INDEX
    tokenizer = Tokenizer(lower=True, char_level=False)
    tokenizer.fit_on_texts(questions)
    WORD_INDEX = tokenizer.word_index

    questions_encoded = tokenizer.texts_to_sequences(questions)
    questions_encoded_padded = pad_sequences(questions_encoded, maxlen=15, padding='post')

    for i, ele in enumerate(tags):
        for j, tag in enumerate(ele):
            if len(tag) == 0 or tag == ',':
                del tags[i][j]

    global  LABEL_ENCODER
    encoder = MultiLabelBinarizer()
    encoder.fit(tags)
    LABEL_ENCODER = encoder
    tags_encoded = encoder.fit_transform(tags)
    LABEL_COUNT = len(tags_encoded[0])  # No. of labels
    print("\tUnique Tokens in Training Data: ", len(WORD_INDEX))
    print("\nNumber of labels: ", LABEL_COUNT)

    with open('./saved/trec_label_encoder.pkl', 'wb') as f:
        pickle.dump(LABEL_ENCODER, f)

    with open('./saved/trec_word_index.pkl', 'wb') as f:
        pickle.dump(WORD_INDEX, f)

    with open('./saved/trec_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    return questions_encoded_padded, tags_encoded

loader_encoder('trec_big')



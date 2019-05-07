import re
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import keras
import tensorflow as tf
from keras.layers import  Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Reshape, Concatenate, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM
from keras.models import Model, load_model
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.backend import manual_variable_initialization
import mysql.connector

import pickle
class OutputObserver(Callback):

    def __init__(self, model, classifier, table):
        self.model = model
        self.classifier = classifier
        self.table = table
    def on_epoch_end(self, epoch, logs={}):

        if self.table=='test':
            #sensor (photography),
            self.classifier.tag_question(self.model, "Is this sensor ghosting, or something else?")

            #colour (photography),
            self.classifier.tag_question(self.model, "The reason for my pale colored / bad contrast film images?")

            #camera (photography),lens (photography),
            self.classifier.tag_question(self.model, "Cameras using mirrors instead of lenses to get coloured images?")

        if self.table=='trec':

            self.classifier.tag_question(self.model, "Who was the king of the Chinese ?")

            self.classifier.tag_question(self.model, "How much do fruit cost there in china ?")

            self.classifier.tag_question(self.model, "Who was the king of the Chinese ? How much do fruit cost in China ?")

            self.classifier.tag_question(self.model, "Who was the king of the Chinese and how much do fruits cost there in china ?")

            self.classifier.tag_question(self.model, "How To download images from Internet and what's the term for chinese fruits ?")

            self.classifier.tag_question(self.model, "Where is India located?")


class SentenceClassifier:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 55
        self.EMBEDDING_DIM = 100
        self.LABEL_COUNT = 0
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None

    def clean_str(self, string):
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

    def loader_encoder(self, table, type="json"):
        """
        Load and encode data from dataset.

        type = "sql" means get data from MySQL database.
        type = "json" means get data from .json file.        """

        if type == "sql":
            mydb, cursor = self.connect_to_db()

            cursor.execute("select question from " + table)  # load questions from db
            questions = list(str(x[0]) for x in cursor.fetchall())

            cursor.execute("select tags from " + table)
            tags = list(re.split(',\s*', tag[0]) for tag in cursor.fetchall())

            del (mydb)
            del (cursor)

        if type == "json":
            with open('./data/' + table + '.json', 'r', encoding='utf8') as f:
                datastore = json.load(f)
                questions = []
                tags = []
                for row in datastore:
                    questions.append(self.clean_str(row['question']))
                    tags.append(row['tags'].split(','))

        if table=='trec' and os.path.exists('./saved/trec_tokenizer.pkl'):

            with open('./saved/trec_tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)

                self.WORD_INDEX = tokenizer.word_index
        else:
            tokenizer = Tokenizer(lower=True, char_level=False)
            tokenizer.fit_on_texts(questions)
            self.WORD_INDEX = tokenizer.word_index

        questions_encoded = tokenizer.texts_to_sequences(questions)
        questions_encoded_padded = pad_sequences(questions_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')


        for i, ele in enumerate(tags):
            for j, tag in enumerate(ele):
                if len(tag) == 0 or tag == ',':
                    del tags[i][j]

        if table=='trec' and os.path.exists('./saved/trec_label_encoder.pkl'):
            with open('./saved/trec_label_encoder.pkl', 'rb') as f:
                self.LABEL_ENCODER = pickle.load(f)
                self.LABEL_COUNT = 6
                encoder = self.LABEL_ENCODER
                tags_encoded = encoder.fit_transform(tags)
        else:
            encoder = MultiLabelBinarizer()
            encoder.fit(tags)
            self.LABEL_ENCODER = encoder
            tags_encoded = encoder.fit_transform(tags)
            self.LABEL_COUNT = len(tags_encoded[0]) #No. of labels
        print("\tUnique Tokens in Training Data: ", len(self.WORD_INDEX))
        print("\nNumber of labels: ", self.LABEL_COUNT)
        #print("*** ", self.LABEL_ENCODER.classes_)
        return questions_encoded_padded, tags_encoded

    def load_embeddings(self, EMBED_PATH='./embeddings/glove.6B.100d.txt'):
        """
        Load pre-trained embeddings into memory.
        """
        embeddings_index = {}
        try:
        	f = open(EMBED_PATH, encoding='utf-8')
        except FileNotFoundError:
        	print("Embeddings missing.")
        	sys.exit()
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
        f.close()
        print("\tNumber of tokens in embeddings file: ", len(embeddings_index))
        return embeddings_index

    def create_embedding_matrix(self, embeddings_index):
        """
        Creates an embedding matrix for all the words(vocab) in the training data with shape (vocab, EMBEDDING_DIM).
        Out-of-vocab words will be randomly initialized to values between +0.25 and -0.25.
        """
        words_not_found = []
        vocab = len(self.WORD_INDEX) + 1
        embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab, self.EMBEDDING_DIM))
        for word, i in self.WORD_INDEX.items():
            if i >= vocab:
                continue
            embedding_vector = embeddings_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)

        print("\tShape of embedding matrix: ", str(embedding_matrix.shape))
        print("\tNo. of words not found in pre-trained embeddings: ", len(words_not_found))
        return embedding_matrix

    def sentence_classifier(self, embedding_matrix, x, y, table, load_saved=0):
        """

        Makes uses of Keras functional API for constructing the model.

        If load_saved=1, THEN load old model, ELSE train new model
        """

        model_name = table + ".model.h5"
        if load_saved == 1 and os.path.exists('./saved/' + model_name):

            print("\nLoading saved model:" + model_name )
            model = load_model('./saved/' + model_name)
            print("Model Summary")
            print(model.summary())

        else:
            print("\nTraining model...")
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)

            X = keras.layers.SpatialDropout1D(0.2)(embedding)


            output = Dense(units=self.LABEL_COUNT, activation='sigmoid')(X)

            model = Model(inputs=inputs, outputs=output, name='question_classifier')
            print("Model Summary")
            print(model.summary())

            cbk = OutputObserver(model, self, table)
            adam = keras.optimizers.Adam(lr=1e-5, decay=1e-6, epsilon=1e-7)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x, y,
                      batch_size=200,
                      epochs=800,
                      verbose=2,
                      callbacks=[cbk])

        return model

    def tag_question(self, model, question, graph=None):

        question = self.clean_str(question)
        print(question)
        question_encoded = [[self.WORD_INDEX[w] for w in question.split(' ') if w in self.WORD_INDEX]]
        question_encoded_padded = pad_sequences(question_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        predictions = model.predict(question_encoded_padded)


        possible_tags = []
        for i, probability in enumerate(predictions[0]):
            if probability >= 0.05:
                possible_tags.append([self.LABEL_ENCODER.classes_[i], probability])

        possible_tags.sort(reverse=True, key=lambda x:x[1]) #sort in place on the basis of the probability in each sub-list in descending order
        print(possible_tags)
        return possible_tags


    def setup_classifier(self, table="trec", load_saved=0):
        '''

        '''

        keras.backend.clear_session()
        print("Loading Data Set...")
        x, y = self.loader_encoder(table)

        embeddings_index = self.load_embeddings()

        print("\nGenerating embedding matrix...")
        embedding_matrix = self.create_embedding_matrix(embeddings_index)

        #Loading / Training model
        model = self.sentence_classifier(embedding_matrix, x, y, table, load_saved=load_saved)

        return model, embeddings_index

    def connect_to_db(self):
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
        cursor = mydb.cursor()
        return mydb, cursor

if __name__ == '__main__':

    classifier = SentenceClassifier()
    model, embeddings_index = classifier.setup_classifier('trec')

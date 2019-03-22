import re
import os
import sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Reshape, concatenate, Concatenate, Flatten
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAvgPool1D, Embedding, Dropout, LSTM
from keras.models import Model, load_model

import mysql.connector


class SentenceClassifier:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 55
        self.EMBEDDING_DIM = 50
        self.LABEL_COUNT = 0
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None

    def clean_str(self, string):
        """
        Cleans each string and convert to lower case.
        """

        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", " will", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def loader_encoder(self):
        """
        Load and encode data from database.
        """
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
        cursor = mydb.cursor()

        cursor.execute("select question from questions") #load questions from db
        results = list(str(x) for x in cursor.fetchall())

        tokenizer = Tokenizer(lower=True, char_level=False)
        tokenizer.fit_on_texts(results)
        self.WORD_INDEX = tokenizer.word_index

        questions_encoded = tokenizer.texts_to_sequences(results)
        questions_encoded_padded = pad_sequences(questions_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        cursor.execute("select tags from questions") #load tags from db
        results = list(x for x in cursor.fetchall())

        encoder = MultiLabelBinarizer()
        encoder.fit(results)
        self.LABEL_ENCODER = encoder
        tags_encoded = encoder.fit_transform(results)
        tags_encoded = encoder.fit_transform(results)
        self.LABEL_COUNT = len(tags_encoded[0]) #No. of labels
        print("\tUnique Tokens in Training Data: ", len(self.WORD_INDEX))
        return questions_encoded_padded, tags_encoded

    def load_embeddings(self, EMBED_PATH='./embeddings/glove.6B.50d.txt'):
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
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("\tShape of embedding matrix: ", str(embedding_matrix.shape))
        print("\tNo. of words not found in pre-trained embeddings: ", len(words_not_found))
        return embedding_matrix

    def get_conv_pool(self, input, n_grams=[3, 5, 7], pool_dim=[48, 47, 46], feature_maps=128):
        '''
        Creates different convolutional layer branches.
        '''
        branches = []
        for i in range(len(n_grams)):
            branch = keras.layers.Conv2D(128, kernel_size=(n_grams[i], self.EMBEDDING_DIM), padding='valid', kernel_initializer='normal',
                    activation='relu',)(input)
            branch = keras.layers.MaxPool2D(pool_size=(pool_dim[i], 1), strides=(1, 1), padding='valid')(branch)

            branches.append(branch)
        return branches

    def sentence_classifier_cnn(self, embedding_matrix, load_saved=0):
        """
        A static CNN model.
        Makes uses of Keras functional API for constructing the model.

        If load_saved=1, THEN load old model, ELSE train new model
        """

        if load_saved == 1 and os.path.exists('./saved/q_tagger.model.h5'):
            print("\nLoading saved model...")
            model = load_model('./saved/q_tagger.model.h5')

        else:
            print("\nTraining model...")
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)

            conv_inputs = Reshape((self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_DIM, 1))(embedding)

            branches = self.get_conv_pool(conv_inputs)
            convolved_tensor = Concatenate(axis=1)(branches)

            flatten = Flatten()(convolved_tensor)
            dropout = Dropout(0.5)(flatten)

            gru = keras.layers.GRU(50, return_sequences=False,)(dropout)
            #lstm = LSTM(64, return_sequences=True)(dropout)
            hidden = Dense(units=500, activation="relu")(gru)
            output = Dense(units=self.LABEL_COUNT, activation='sigmoid', name='fully_connected_affine_layer')(hidden)

            model = Model(inputs=inputs, outputs=output, name='intent_classifier')
            print("Model Summary")
            print(model.summary())

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x, y,
                      batch_size=65,
                      epochs=20,
                      verbose=2)

            model.save('./saved/q_tagger.model.h5')


        return model


if __name__ == '__main__':
    classifier = SentenceClassifier()

    print("Loading Data set...")
    x,y = classifier.loader_encoder()

    embeddings_index = classifier.load_embeddings()

    print("Generating Embedding Matrix...")
    embedding_matrix = classifier.create_embedding_matrix(embeddings_index)
    model = classifier.sentence_classifier_cnn(embedding_matrix)

    sentence = ["Who killed him and what is the costliest disaster the industry has ever faced?"]
    sentence = classifier.clean_str(sentence[0])
    sentence_encoded = [[classifier.WORD_INDEX[w] for w in sentence.split(' ') if w in classifier.WORD_INDEX]]

    sentence_encoded_padded = pad_sequences(sentence_encoded, maxlen=classifier.MAX_SEQUENCE_LENGTH, padding='post')
    pred = model.predict(sentence_encoded_padded)

    print(pred)
    for i, probability in enumerate(pred[0]):
        if probability >= 0.1:
            print(classifier.LABEL_ENCODER.classes_[i], " ", probability)

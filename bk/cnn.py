import re
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Reshape, concatenate, Concatenate, Flatten
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAvgPool1D, Embedding, Dropout, LSTM
from keras.models import Model, load_model

import mysql.connector


class SentenceClassifier:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 200
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

    def loader_encoder(self, table, type="sql"):
        """
        Load and encode data from dataset.

        type = "sql" means get data from MySQL database.
        type = "json" means get data from .json file.
        """

        if type == "sql":
            mydb, cursor = self.connect_to_db()
            cursor.execute("select question from " + table) #load questions from db
            questions = list(str(x) for x in cursor.fetchall())
            cursor.execute("select tags from " + table)  # load tags from db
            tags = list(re.split(',\s*', tag[0]) for tag in cursor.fetchall())

        elif type == "json":
            with open('./data/stack-exchange/Law.json', 'r', encoding='utf8') as f:
                datastore = json.load(f)
                questions = []
                tags = []
                for row in datastore:
                    questions.append(row['question'])
                    tags.append(row['tags'].split(', '))

        tokenizer = Tokenizer(lower=True, char_level=False)
        tokenizer.fit_on_texts(questions)
        self.WORD_INDEX = tokenizer.word_index

        questions_encoded = tokenizer.texts_to_sequences(questions)
        questions_encoded_padded = pad_sequences(questions_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        for i, ele in enumerate(tags):
            for j, tag in enumerate(ele):
                if len(tag) == 0 or tag == ',':
                    del tags[i][j]

        encoder = MultiLabelBinarizer()
        encoder.fit(tags)
        self.LABEL_ENCODER = encoder
        tags_encoded = encoder.fit_transform(tags)
        self.LABEL_COUNT = len(tags_encoded[0]) #No. of labels
        print("\tUnique Tokens in Training Data: ", len(self.WORD_INDEX))

        del(mydb)
        del(cursor)

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
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("\tShape of embedding matrix: ", str(embedding_matrix.shape))
        print("\tNo. of words not found in pre-trained embeddings: ", len(words_not_found))
        return embedding_matrix

    def get_top_k(self, x):
        shifted_input = tf.transpose(x, [0, 2, 1])
        top_k = tf.nn.top_k(x, k=2, sorted=True)[0]
        return Flatten()(top_k)

    def get_conv_pool(self, input, n_grams=[3, 5, 7]):
        '''
        Creates different convolutional layer branches.
        '''
        branches = []
        for i in range(len(n_grams)):
            branch = keras.layers.Conv1D(128, kernel_size=n_grams[i], padding='valid',kernel_initializer='normal', activation='relu', )(input)

            branches.append(branch)
        return branches

    def sentence_classifier_cnn(self, embedding_matrix, x, y, table, load_saved=1):
        """
        A static CNN model.
        Makes uses of Keras functional API for constructing the model.

        If load_saved=1, THEN load old model, ELSE train new model
        """

        model_name = table + ".model.h5"
        if load_saved == 1 and os.path.exists('./saved/' + model_name):
            print("\nLoading saved model...")
            model = load_model('./saved/' + model_name)

        else:
            print("\nTraining model...")
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)

            dropout = Dropout(0.2)(embedding)

            main = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(dropout)
            main = keras.layers.MaxPooling1D(pool_size=2)(main)
            main = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(main)
            main = keras.layers.MaxPooling1D(pool_size=2)(main)
            main = keras.layers.GRU(64)(main)
            main = Dense(200, activation="relu")(main)
            main = Dense(self.LABEL_COUNT, activation="sigmoid")(main)
            #conv_inputs = Reshape((self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_DIM))(embedding)

            #branches = self.get_conv_pool(conv_inputs)
            #convolved_tensor = Concatenate(axis=1)(branches)

            #convolved_tensor = keras.layers.Lambda(self.get_top_k)(convolved_tensor)

            #flatten = Flatten()(convolved_tensor)


            #hidden = Dense(units=500, activation="relu")(convolved_tensor)
            #output = Dense(units=self.LABEL_COUNT, activation='sigmoid')(hidden)

            model = Model(inputs=inputs, outputs=main, name='intent_classifier')
            print("Model Summary")
            print(model.summary())

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x, y,
                      batch_size=65,
                      epochs=15,
                      verbose=2)

            model.save('./saved/' + model_name)
        #keras.utils.vis_utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model




    def tag_question(self, model, question):

        question = self.clean_str(question)
        question_encoded = [[self.WORD_INDEX[w] for w in question.split(' ') if w in self.WORD_INDEX]]
        question_encoded_padded = pad_sequences(question_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        predictions = model.predict(question_encoded_padded)

        possible_tags = []
        for i, probability in enumerate(predictions[0]):
            if probability >= 0.1:
                possible_tags.append([self.LABEL_ENCODER.classes_[i], probability])

        possible_tags.sort(reverse=True, key=lambda x:x[1]) #sort in place on the basis of the probability in each sub-list in descending order
        print(possible_tags)
        return possible_tags

    def setup_classifier(self, table):
        '''

        '''
        print("Loading Data Set...")
        x, y = self.loader_encoder(table)

        embeddings_index = self.load_embeddings()

        print("\nGenerating embedding matrix...")
        embedding_matrix = self.create_embedding_matrix(embeddings_index)

        #Loading / Training model
        model = self.sentence_classifier_cnn(embedding_matrix, x, y, table, load_saved=1)

        return model, embeddings_index

    def connect_to_db(self):
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
        cursor = mydb.cursor()
        return mydb, cursor

if __name__ == '__main__':
    classifier = SentenceClassifier()
    model, embeddings_index = classifier.setup_classifier('compiled')
    sentence = "What's the difference between copyright and trademark?"
    classifier.tag_question(model, sentence)


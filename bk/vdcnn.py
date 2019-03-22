import re
import os
import sys
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

    def loader_encoder(self, table):
        """
        Load and encode data from database.
        """
        mydb, cursor = self.connect_to_db()

        cursor.execute("select question from " + table) #load questions from db
        results = list(str(x) for x in cursor.fetchall())

        tokenizer = Tokenizer(lower=True, char_level=False)
        tokenizer.fit_on_texts(results)
        self.WORD_INDEX = tokenizer.word_index

        questions_encoded = tokenizer.texts_to_sequences(results)
        questions_encoded_padded = pad_sequences(questions_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        cursor.execute("select tags from " + table) #load tags from db
        tags = list(re.split(',\s*', tag[0]) for tag in cursor.fetchall())

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

    def get_conv_pool(self, input, n_grams=[2, 2, 3, 3, 4]):
        '''
        Creates different convolutional layer branches.
        '''
        branches = []
        for i in range(len(n_grams)):
            branch = keras.layers.Conv1D(128, kernel_size=n_grams[i], padding='valid',kernel_initializer='normal', activation='relu', )(input)
            branch = keras.layers.GlobalMaxPooling1D()(branch)

            branches.append(branch)
        return branches

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def get_top_k(self, x):
        shifted_input = tf.transpose(x, [0, 2, 1])
        top_k = tf.nn.top_k(x, k=1, sorted=True)[0]
        return Flatten()(top_k)

    def sentence_classifier_cnn(self, embedding_matrix, x, y, table, load_saved=1):
        """
        A static CNN model.
        Makes uses of Keras functional API for constructing the model.

        If load_saved=1, THEN load old model, ELSE train new model
        """

        model_name = table + ".model.h5"
        if load_saved == 1 and os.path.exists('./saved/' + model_name):
            print("\nLoading saved model...")
            model = load_model('./saved/' + model_name, custom_objects={
                                                        'tf':tf,

                                                    })

        else:
            print("\nTraining model...")
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)

            num_filters = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]

            conv = embedding
            for i in range(len(num_filters)):
                conv = keras.layers.Conv1D(filters=num_filters[i], kernel_size=3, padding='same')(conv)
                conv = keras.layers.BatchNormalization()(conv)
                conv = keras.layers.Activation('relu')(conv)
                conv = keras.layers.Conv1D(filters=num_filters[i], kernel_size=3, padding='same')(conv)
                conv = keras.layers.BatchNormalization()(conv)
                conv = keras.layers.Activation('relu')(conv)
                conv = keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(conv)


            k_max = keras.layers.Lambda(self.get_top_k)(conv)

            # 3 fully-connected layer with dropout regularization
            fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
            fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
            fc3 = Dense(self.LABEL_COUNT, activation='sigmoid')(fc2)





            model = Model(inputs=inputs, outputs=fc3, name='intent_classifier')
            print("Model Summary")
            print(model.summary())

            sgd = keras.optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=False)
            model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
            model.fit(x, y,
                      batch_size=65,
                      epochs=25,
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
    classifier.tag_question(model, "If I violated copyright and trademark illegal, would that  not be unconstitutional by law?")

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
from keras.engine.topology import Layer
from keras import  backend as K
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras_self_attention import SeqSelfAttention
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

    def get_conv_pool(self, input, n_grams=[3]):
        '''
        Creates different convolutional layer branches.
        '''
        branches = []
        for i in range(len(n_grams)):
            branch = keras.layers.Conv1D(128, kernel_size=n_grams[i], padding='valid',kernel_initializer='normal', activation='relu', )(input)
            branch = keras.layers.GlobalMaxPooling1D()(branch)
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
            model = load_model('./saved/' + model_name, custom_objects={"AttentionWithContext": AttentionWithContext})

        else:
            print("\nTraining model...")
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)


            word = keras.layers.SpatialDropout1D(0.2)(embedding)
            word_lstm = keras.layers.Bidirectional(LSTM(150, return_sequences=True, ))(word)
            word_dense = keras.layers.TimeDistributed(Dense(units=200, activation="relu"))(word_lstm)
            word_out = AttentionWithContext()(word_dense)
            preds = keras.layers.Dense(units = self.LABEL_COUNT, activation='sigmoid')(word_out)

            model = Model(inputs=inputs, outputs=preds, name='intent_classifier')
            print("Model Summary")
            print(model.summary())

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x, y,
                      batch_size=65,
                      epochs=30,
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

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def dot_product(self, x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

    def call(self, x, mask=None):
        uit = self.dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = self.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

if __name__ == '__main__':
    classifier = SentenceClassifier()
    model, embeddings_index = classifier.setup_classifier('compiled')
    classifier.tag_question(model, "What is the difference between trademark and copyright will violation take me to court since I've broken the constitution law?")

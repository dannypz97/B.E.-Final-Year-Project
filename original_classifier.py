import os
import re
import sys
from time import time
from datetime import timedelta
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, Reshape, Concatenate, Flatten
from keras.layers import Conv2D, MaxPool2D, Embedding, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

class SentenceClassifier:
    def __init__(self):
        self.DATA_DIR = './data/trec/'
        self.EMBED_DIR = './embeddings/'
        self.MAX_SEQUENCE_LENGTH = 55
        self.EMBEDDING_DIM = 100
        self.LABEL_COUNT = 0
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None
        self.VALIDATION_SPLIT = 0.2

    def train_dev_split(self, X_train_encoded_padded, Y_train_one_hot):

        indices = np.arange(X_train_encoded_padded.shape[0])
        np.random.shuffle(indices)
        X_train_encoded_padded = X_train_encoded_padded[indices]
        Y_train_one_hot = Y_train_one_hot[indices]
        num_validation_samples = int(self.VALIDATION_SPLIT * X_train_encoded_padded.shape[0])
        x_train = X_train_encoded_padded[:-num_validation_samples]
        y_train = Y_train_one_hot[:-num_validation_samples]
        x_val = X_train_encoded_padded[-num_validation_samples:]
        y_val = Y_train_one_hot[-num_validation_samples:]
        return x_train, y_train, x_val, y_val

    def clean_str(self, string):

        string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
        string = re.sub(r" : ", ":", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_train(self):

        data_set = list(open(self.DATA_DIR + 'train_5500.label.txt', encoding='utf-8', errors='replace').readlines())
        data_set_cleaned = [self.clean_str(sent) for sent in data_set]
        Y_Train = [s.split(' ')[0].split(':')[0] for s in data_set_cleaned]
        X_Train = [s.split(" ")[1:] for s in data_set_cleaned]
        return X_Train, Y_Train

    def load_data_test(self):

        data_set = list(open(self.DATA_DIR + 'TREC_10.label.txt', encoding='utf-8', errors='replace').readlines())
        data_set_cleaned = [self.clean_str(sent) for sent in data_set]
        Y_Test = [s.split(' ')[0].split(':')[0] for s in data_set_cleaned]
        X_Test = [s.split(" ")[1:] for s in data_set_cleaned]
        return X_Test, Y_Test

    def integer_encode_train(self, X_Train, Y_Train):

        tokenizer = Tokenizer(lower=True, char_level=False)
        tokenizer.fit_on_texts(X_Train)
        self.WORD_INDEX = tokenizer.word_index
        X_train_encoded = tokenizer.texts_to_sequences(X_Train)
        X_train_encoded_padded = pad_sequences(X_train_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        encoder = LabelEncoder()
        encoder.fit(Y_Train)
        self.LABEL_ENCODER = encoder
        Y_train_encoded = encoder.transform(Y_Train)
        Y_train_one_hot = np_utils.to_categorical(Y_train_encoded)
        self.LABEL_COUNT = Y_train_one_hot.shape[1]
        print("\tUnique Tokens in Training Data: %s" % len(self.WORD_INDEX))
        print("\tShape of data tensor (X_train): %s" % str(X_train_encoded_padded.shape))
        print("\tShape of label tensor (Y): %s" % str(Y_train_one_hot.shape))
        return X_train_encoded_padded, Y_train_one_hot

    def integer_encode_test(self, X_Test, Y_Test):

        X_test_encoded = list()
        for sentence in X_Test:
            x_test = [self.WORD_INDEX[w] for w in sentence if w in self.WORD_INDEX]
            X_test_encoded.append(x_test)
        X_test_encoded_padded = pad_sequences(X_test_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        Y_test_encoded = self.LABEL_ENCODER.transform(Y_Test)
        Y_test_one_hot = np_utils.to_categorical(Y_test_encoded)
        print("\tUnique Tokens in Test Data (this should be same as in Training Data): %s" % len(self.WORD_INDEX))
        print("\tShape of data tensor (X_test): %s" % str(X_test_encoded_padded.shape))
        print("\tShape of label tensor (Y_test): %s" % str(Y_test_one_hot.shape))
        return X_test_encoded_padded, Y_test_one_hot



    def load_glove(self):

        embeddings_index = {}
        try:
        	f = open(os.path.join(self.EMBED_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
        except FileNotFoundError:
        	print("GloVe vectors missing. You can download from http://nlp.stanford.edu/data/glove.6B.zip")
        	sys.exit()
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print("\tNumber of Tokens from GloVe: %s" % len(embeddings_index))
        return embeddings_index

    def glove_embedding_matrix(self, embeddings_index):

        words_not_found = []
        vocab = len(self.WORD_INDEX) + 1
        # embedding_matrix = np.zeros((vocab, self.EMBEDDING_DIM))
        embedding_matrix = np.random.uniform(-0.1, 0.1, size=(vocab, self.EMBEDDING_DIM))  # 0.25 is chosen so
        # the unknown vectors have (approximately) same variance as pre-trained ones
        for word, i in self.WORD_INDEX.items():
            if i >= vocab:
                continue
            embedding_vector = embeddings_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("\tShape of embedding matrix: %s" % str(embedding_matrix.shape))
        print("\tNo. of words not found in GloVe: ", len(words_not_found))
        return embedding_matrix

    def sentence_classifier_cnn(self, embedding_matrix):

        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        X = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)(inputs)


        X = keras.layers.SpatialDropout1D(0.2)(X)

        X1 = keras.layers.Conv1D(36, kernel_size=1, activation="relu")(X)
        X1 = keras.layers.Conv1D(36, kernel_size=2, activation="relu")(X1)
        X1 = keras.layers.GlobalMaxPooling1D()(X1)

        X2 = keras.layers.Conv1D(36, kernel_size=1, activation="relu")(X)
        X2 = keras.layers.Conv1D(36, kernel_size=2, activation="relu")(X2)
        X2 = keras.layers.Conv1D(36, kernel_size=2, activation="relu")(X2)
        X2 = keras.layers.GlobalMaxPooling1D()(X2)


        X = Concatenate(axis=1)([X1, X2])
        X = keras.layers.BatchNormalization()(X)
        output = Dense(units=self.LABEL_COUNT, activation='softmax', name='fully_connected_affine_layer')(X)

        model = Model(inputs=inputs, outputs=output, name='intent_classifier')
        print("Model Summary")
        print(model.summary())
        adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=30,
                  validation_data=(x_val, y_val), verbose=2)

        plt.plot(history.history['acc'], linestyle='-')
        plt.plot(history.history['val_acc'], linestyle='--')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'], linestyle='-')
        plt.plot(history.history['val_loss'], linestyle='--')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #plot_model(model, to_file='sentence_classifier_cnn.png')
        return model


if __name__ == '__main__':
    sampleClassifier = SentenceClassifier()
    print("Loading Training Data set...")
    X_Train, Y_Train = sampleClassifier.load_data_train()
    print("Encoding Training Data set...")
    X_train_encoded_padded, Y_train_one_hot = sampleClassifier.integer_encode_train(X_Train, Y_Train)
    print("Splitting Data set to Train and Validation set...")
    x_train, y_train, x_val, y_val = sampleClassifier.train_dev_split(X_train_encoded_padded, Y_train_one_hot)
    #x_train, y_train = X_train_encoded_padded, Y_train_one_hot
    print("Loading GloVe vectors...")
    embeddings_index = sampleClassifier.load_glove()
    print("Generating Embedding Matrix...")
    embedding_matrix = sampleClassifier.glove_embedding_matrix(embeddings_index)

    model = sampleClassifier.sentence_classifier_cnn(embedding_matrix)

    # Evaluating the model
    print("Evaluating the model...")
    print("Loading Test Data set...")
    X_Test, Y_Test = sampleClassifier.load_data_test()
    print("Encoding Test Data set...")
    X_test_encoded_padded, Y_test_one_hot = sampleClassifier.integer_encode_test(X_Test, Y_Test)
    print("Evaluating the model on the Test Data set...")
    scores = model.evaluate(X_test_encoded_padded, Y_test_one_hot, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
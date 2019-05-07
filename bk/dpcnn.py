import re
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks

import mysql.connector


class SentenceClassifier:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 200
        self.EMBEDDING_DIM = 100
        self.LABEL_COUNT = 0
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None
        self.tag_dict = dict()  #maps broad labels to associated fine labels

    def clean_str(self, string):
        """
        Cleans each string and convert to lower case.
        """
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
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

            '''
            sql ="select `broad_label`, `fine_label` from tags where `table` = %s"
            cursor.execute(sql, (table,))
            tag_rows = cursor.fetchall()
            for ele in tag_rows:
                if ele[0].lower() not in self.tag_dict:
                    self.tag_dict[ele[0].lower()] = []
                if ele[1].isupper():
                    self.tag_dict[ele[0].lower()].append(ele[1])
                else:
                    self.tag_dict[ele[0].lower()].append(ele[1].lower())
            '''

            del (mydb)
            del (cursor)

        if type == "json":
            with open('data/' + table + '.json', 'r', encoding='utf8') as f:
                datastore = json.load(f)
                questions = []
                tags = []
                for row in datastore:
                    questions.append(row['question'])
                    tags.append(row['tags'].split(','))

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

    def sentence_classifier_cnn(self, embedding_matrix, x, y, table, load_saved=1):
        """

        Makes uses of Keras functional API for constructing the model.

        If load_saved=1, THEN load old model, ELSE train new model
        """
        filter_nr = 64
        filter_size = 3
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 256
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)
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

            X = SpatialDropout1D(0.2)(embedding)

            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(X)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)

            # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
            # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
            resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(X)
            resize_emb = PReLU()(resize_emb)

            block1_output = add([block1, resize_emb])
            block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
            block2 = BatchNormalization()(block2)
            block2 = PReLU()(block2)
            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
            block2 = BatchNormalization()(block2)
            block2 = PReLU()(block2)

            block2_output = add([block2, block1_output])
            block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

            block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
            block3 = BatchNormalization()(block3)
            block3 = PReLU()(block3)
            block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
            block3 = BatchNormalization()(block3)
            block3 = PReLU()(block3)

            block3_output = add([block3, block2_output])
            block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

            block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
            block4 = BatchNormalization()(block4)
            block4 = PReLU()(block4)
            block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
            block4 = BatchNormalization()(block4)
            block4 = PReLU()(block4)

            block4_output = add([block4, block3_output])
            block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

            block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
            block5 = BatchNormalization()(block5)
            block5 = PReLU()(block5)
            block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
            block5 = BatchNormalization()(block5)
            block5 = PReLU()(block5)

            block5_output = add([block5, block4_output])
            block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

            block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
            block6 = BatchNormalization()(block6)
            block6 = PReLU()(block6)
            block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
            block6 = BatchNormalization()(block6)
            block6 = PReLU()(block6)

            block6_output = add([block6, block5_output])
            block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

            block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
            block7 = BatchNormalization()(block7)
            block7 = PReLU()(block7)
            block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
            block7 = BatchNormalization()(block7)
            block7 = PReLU()(block7)

            block7_output = add([block7, block6_output])
            output = GlobalMaxPooling1D()(block7_output)

            output = Dense(dense_nr, activation='linear')(output)
            output = BatchNormalization()(output)
            output = PReLU()(output)
            output = Dropout(dense_dropout)(output)
            output = Dense(self.LABEL_COUNT, activation='sigmoid')(output)

            model = Model(inputs=inputs, outputs=output, name='intent_classifier')
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

        print(len(self.LABEL_ENCODER.classes_))
        print(len(predictions[0]))

        possible_tags = []
        for i, probability in enumerate(predictions[0]):
            if probability >= 0.01:
                possible_tags.append([self.LABEL_ENCODER.classes_[i], probability])

        possible_tags.sort(reverse=True, key=lambda x:x[1]) #sort in place on the basis of the probability in each sub-list in descending order
        print(possible_tags)
        return possible_tags


    def setup_classifier(self, table="compiled"):
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
    model, embeddings_index = classifier.setup_classifier('trec')
    classifier.tag_question(model, "Who was the prophet of the Muslim people and where is India located and what does booty mean to the homies?")
    classifier.tag_question(model, "Where is India located and Who was the prophet of the Muslim people ?")

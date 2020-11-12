import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from konlpy.tag import Okt
import numpy as np
import pandas as pd
import argparse

# setting hyper-parameters ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--sequence_length', type=int, default=150)
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--filter_sizes', type=str, default='3,4,5')
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--regularizers_lambda', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()
########################################################################

def make_morphs_list(documents):
    okt=Okt()
    morphs_list = []
    for i, document in enumerate(documents):
        clean_words = []
        for word in okt.morphs(document):
            clean_words.append(word)
        document = ' '.join(clean_words)
        morphs_list.append(document)

    return morphs_list

def make_morphs_list(documents):
    okt=Okt()
    morphs_list = []
    for i, document in enumerate(documents):
        clean_words = []
        for word in okt.morphs(document):
            clean_words.append(word)
        document = ' '.join(clean_words)
        morphs_list.append(document)

    return morphs_list


PADDING = "<PADDING>"  # Zero padding
OOV = "<OOV>"  # Out Of Vocabulary


def make_dictionary(documents):
    words = []

    for document in documents:
        for word in document.split():
            words.append(word)

    # 길이가 0인 단어를 삭제합니다.
    words = [word for word in words if len(word) > 0]

    # 중복된 단어를 삭제합니다.
    words = list(set(words))

    # 단어 사전의 제일 앞에 태그 단어를 삽입합니다.
    # 단어 사전에서 PADDING의 인덱스는 0, OOV의 인덱스는 1이 됩니다.
    words = [PADDING, OOV] + words

    vocab_size = len(words)

    # 단어 사전 word_to_index를 생성합니다.
    word_to_index = {word: index for index, word in enumerate(words)}

    return word_to_index, vocab_size

def convert_word_to_index(documents, word_to_index):
    documents_index = []

    # 모든 문장에 대해서 반복합니다.
    for document in documents:
        document_index = []

        # 문장의 단어들을 띄어쓰기로 분리합니다.
        for word in document.split():
            if word_to_index.get(word) is not None:
                # 사전에 있는 단어이면 해당 인덱스를 추가합니다.
                document_index.append(word_to_index[word])
            else:
                # 사전에 없는 단어이면 OOV 인덱스를 추가합니다.
                document_index.append(word_to_index[OOV])

        # 최대 길이(arg.sequence_length)를 확인합니다.
        if len(document_index) > args.sequence_length:
            document_index = document_index[:args.sequence_length]

        # 최대 길이보다 짧은 문장은 남는 토큰(최대 길이-문장의 토큰 개수)만큼을 PADDING 인덱스로 채웁니다.
        document_index += [word_to_index[PADDING]] * (args.sequence_length - len(document_index))

        # 문장의 인덱스 배열을 document_index 리스트에 추가합니다.
        documents_index.append(document_index)

    return np.asarray(documents_index)

def label_one_hot_encoding(label):
    results = np.zeros((len(label), 2))
    for idx, l in enumerate(label):
        if l == 0: # negative
            results[idx] = [1, 0]
        else: # positive
            results[idx] = [0, 1]
    return results

# loading data from filestorage directory ##############################
df = pd.read_csv("filestorage/ratings_preprocessed.csv", sep='\t')
########################################################################
documents = make_morphs_list(df['document'].values)
word_to_index, vocab_size = make_dictionary(documents)
documents_index = convert_word_to_index(documents, word_to_index)
labels = label_one_hot_encoding(df['label'].values)

X_train, Y_train = documents_index[:52263], labels[:52263]
X_test, Y_test = documents_index[52263:], labels[52263:]

def TextCNN(vocab_size, sequence_length, embed_size, num_classes, num_filters,
            filter_sizes, regularizers_lambda, dropout_rate):

    inputs = keras.Input(shape=(sequence_length,), name='input_data')
    embed = keras.layers.Embedding(vocab_size, embed_size,
                                   embeddings_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
                                   input_length=sequence_length,
                                   name='embedding')(inputs)

    embed = keras.layers.Reshape((sequence_length, embed_size, 1), name='add_channel')(embed)

    pool_outputs = []
    for filter_size in list(map(int, filter_sizes.split(','))):
        filter_shape = (filter_size, embed_size)
        conv = keras.layers.Conv2D(num_filters, filter_shape, strides=(1, 1), padding='valid',
                                   data_format='channels_last', activation='relu',
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1),
                                   name='convolution_{:d}'.format(filter_size))(embed)
        max_pool_shape = (sequence_length - filter_size + 1, 1)
        pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                      strides=(1, 1), padding='valid',
                                      data_format='channels_last',
                                      name='max_pooling_{:d}'.format(filter_size))(conv)
        pool_outputs.append(pool)

    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
    pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
    pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(pool_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = TextCNN(vocab_size, args.sequence_length, args.embed_size, 2,
                args.num_filters, args.filter_sizes, args.regularizers_lambda, args.dropout_rate)
model.summary()
model.compile(keras.optimizers.Adam(lr=args.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch {}".format(epoch + 1))
        print("Train-acc={:.4f}".format(logs['accuracy']))
        print("Validation-acc={:.4f}".format(logs['val_accuracy']))

history = model.fit(x=X_train, y=Y_train,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_split=0.25,
                    shuffle=True,
                    callbacks=[MetricHistory()])
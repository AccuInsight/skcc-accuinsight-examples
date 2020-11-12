from tensorflow.keras.layers import Input, Dense, Convolution1D, MaxPool1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from keras.utils.np_utils import to_categorical

import pandas as pd
import argparse

# setting hyper-parameters ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--num_node', type=int, default=64)
args = parser.parse_args()
########################################################################

# loading data from filestorage directory ##############################
train = pd.read_csv("filestorage/mitbih_train.csv", header=None)
test = pd.read_csv("filestorage/mitbih_test.csv", header=None)
########################################################################

train = train.rename({187:'class'}, axis='columns')
test = test.rename({187:'class'}, axis='columns')

y_train = to_categorical(train['class'])
y_test = to_categorical(test['class'])

X_train = train.iloc[:, :186].values
X_test = test.iloc[:, :186].values
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)

class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch {}".format(epoch + 1))
        print("Train-acc={:.4f}".format(logs['accuracy']))
        print("Validation-acc={:.4f}".format(logs['val_accuracy']))

def network(X_train, y_train, X_test, y_test):
    
    im_shape = (X_train.shape[1], 1)
    inputs_cnn = Input(shape=(im_shape), name='inputs_cnn')
    conv1_1 = Convolution1D(args.num_node, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1 = Convolution1D(args.num_node, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    pool2 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1 = Convolution1D(args.num_node, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    pool3 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten = Flatten()(pool3)
    dropout1 = Dropout(args.dropout_rate)(flatten)
    dense_end1 = Dense(args.num_node, activation='relu')(dropout1)
    dropout2 = Dropout(args.dropout_rate)(dense_end1)
    dense_end2 = Dense(args.num_node/2, activation='relu')(dropout2)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
    
    model = Model(inputs=inputs_cnn, outputs=main_output)
    
    model.compile(optimizer=Adam(lr=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 MetricHistory()]

    history = model.fit(X_train, y_train,
                        epochs=args.num_epochs,
                        callbacks=callbacks,
                        batch_size=args.batch_size,
                        validation_data=(X_test,y_test))
    
    return model, history

model, history = network(X_train, y_train, X_test, y_test)
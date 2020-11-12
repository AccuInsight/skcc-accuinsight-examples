import numpy as np
import csv
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn import preprocessing

# setting hyper-parameters ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--num_nodes', type=int, default=50)
parser.add_argument('--l2_penalty', type=float, default=0.001)
args = parser.parse_args()
########################################################################

# loading data from filestorage directory ##############################
train_data_path = "filestorage/creditcard_train.csv"
valid_data_path = "filestorage/creditcard_valid.csv"

feature = np.zeros((255883, 30))
label = np.zeros((255883, 1))
with open(train_data_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for i, row in enumerate(reader):
        label[i, :] = int(row[-1])
        feature[i, :] = row[:30]
########################################################################

input_layer = Input(shape=(feature.shape[1],))

## encoding part
encoded = Dense(args.num_nodes,
                activation='relu',
                activity_regularizer=tf.keras.regularizers.l2(args.l2_penalty))(input_layer)
encoded = Dense(args.num_nodes/2, activation='relu')(encoded)

## decoding part
decoded = Dense(args.num_nodes/2, activation='relu')(encoded)
decoded = Dense(args.num_nodes, activation='relu')(decoded)

## output layer
output_layer = Dense(feature.shape[1], activation='relu')(decoded)

model = Model(input_layer, output_layer)

decay_rate = args.learning_rate / args.epochs
adam = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, decay=decay_rate)
model.compile(optimizer=adam, loss="mse")

feature_scale = preprocessing.MinMaxScaler().fit_transform(feature)

class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch {}".format(epoch + 1))
        print("Train-loss={:.4f}".format(logs['loss']))
        print("Validation-loss={:.4f}".format(logs['val_loss']))
history = MetricHistory()

model.fit(feature_scale,
          feature_scale,
          batch_size = args.batch_size,
          epochs = args.epochs,
          shuffle = True,
          validation_split = 0.2,
          callbacks = [history])
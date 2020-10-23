import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, Input

import numpy as np
import pandas as pd
import argparse
import csv
from datetime import datetime, timezone

# setting hyper-parameters ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--num_nodes', type=int, default=64,
                    help='Number of nodes')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Number of layers')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of samples per gradient update')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to run trainer')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Initial learning rate')
parser.add_argument('--act_func', type=str, default='relu',
                    help='activation function {relu, elu, tanh}')
args = parser.parse_args()
########################################################################

# loading data from filestorage directory ##############################
train = pd.read_csv("filestorage/instagram_train.csv")
test = pd.read_csv("filestorage/instagram_test.csv")
########################################################################

train['profile pic'] = train['profile pic'].astype('category')
train['name==username'] = train['name==username'].astype('category')
train['external URL'] = train['external URL'].astype('category')
train['private'] = train['private'].astype('category')

train_X = pd.get_dummies(train)
train_Y = train_X.pop("fake")

test['profile pic'] = test['profile pic'].astype('category')
test['name==username'] = test['name==username'].astype('category')
test['external URL'] = test['external URL'].astype('category')
test['private'] = test['private'].astype('category')

test_X = pd.get_dummies(test)
test_Y = test_X.pop("fake")

model = tf.keras.Sequential()
model.add(Input(shape=(15,)))
for i in range(args.num_layers):
    model.add(Dense(args.num_nodes, activation=args.act_func))
model.add(Dense(2, activation=args.act_func))

opt = optimizers.Adam(lr=args.learning_rate, amsgrad=True)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\nEpoch {}".format(epoch + 1))
        print("{} Train-acc={:.4f}".format(local_time, logs['accuracy']))
        print("{} Validation-acc={:.4f}".format(local_time, logs['val_accuracy']))

history = MetricHistory()

model.fit(train_X.values, train_Y.values,
          epochs=args.epochs,
          batch_size=args.batch_size,
          validation_data=(test_X, test_Y),
          callbacks=[history])
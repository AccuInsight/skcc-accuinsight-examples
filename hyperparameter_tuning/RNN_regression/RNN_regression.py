import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
from datetime import datetime, timezone

# setting hyper-parameters ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--sequence_length', type=int, default=7)
parser.add_argument('--num_nodes', type=int, default=30)
parser.add_argument('--rnn_cell', type=str, default='LSTM')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of samples per gradient update')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to run trainer')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Initial learning rate')
args = parser.parse_args()
########################################################################

# loading data from filestorage directory ##############################
temp = pd.read_csv("filestorage/seoul_temperature.csv")
########################################################################

temp["날짜"] = pd.to_datetime(temp["날짜"])
temp = temp.set_index("날짜")
temp = pd.DataFrame(temp["평균기온(℃)"])

train_df = temp[:3652]
test_df = temp[3652:]

transformer = MinMaxScaler()
train = transformer.fit_transform(train_df)
test = transformer.transform(test_df)

window_length = args.sequence_length + 1

x_train = []
y_train = []
for i in range(0, len(train) - window_length + 1):
    window = train[i:i + window_length, :]
    x_train.append(window[:-1, :])
    y_train.append(window[-1, [-1]])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for i in range(0, len(test) - window_length + 1):
    window = test[i:i + window_length, :]
    x_test.append(window[:-1, :])
    y_test.append(window[-1, [-1]])
x_test = np.array(x_test)
y_test = np.array(y_test)

if args.rnn_cell == "LSTM":
    rnn_cell = tf.keras.layers.LSTM
elif args.rnn_cell == "GRU":
    rnn_cell = tf.keras.layers.GRU
elif args.rnn_cell == "RNN":
    rnn_cell = tf.keras.layers.SimpleRNN

model = tf.keras.models.Sequential([
    rnn_cell(args.num_nodes, return_sequences=True, input_shape=(args.sequence_length, 1)),
    rnn_cell(args.num_nodes),
    tf.keras.layers.Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.summary()

class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\nEpoch {}".format(epoch + 1))
        print("{} Train-loss={:.4f}".format(local_time, logs['loss']))
        print("{} Validation-loss={:.4f}".format(local_time, logs['val_loss']))

history = MetricHistory()

model.fit(x_train, y_train,
          epochs=args.epochs,
          batch_size=args.batch_size,
          validation_data=(x_test, y_test),
          shuffle=False,
          callbacks=[history])
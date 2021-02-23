#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

class AutoRegressionLSTM(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(units)
        self.lstm_cell_decoder = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        self.dense1 = tf.keras.layers.Dense(8, activation='tanh')  # 시간마다 sales 만 뱉어냄
        self.dense2 = tf.keras.layers.Dense(1, activation='linear') # 시간마다 sales 만 뱉어냄

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        '''
        Output : 
        If return_state: a list of tensors. The first tensor is the output. 
        The remaining tensors are the last states, each with shape [batch_size, state_size], 
        where state_size could be a high dimension tensor shape.
        '''
        # print(x)
        # print(x.shape)  => (# of sample, # of units), x와 state[0]은 동일함(h_t), state[1]은 c_t 의미
        # predictions.shape => (batch, features)

        prediction1 = self.dense1(x)
        prediction = self.dense2(prediction1)

        # prediction = self.dense(x) # seq2seq와 다른 부분. encoder RNN 의 결과로 prediction 시작.
                                   # seq2seq는 encoding 모두 완료하고, decoder에서 prediction 시작
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        
        # Insert the first prediction
        predictions.append(prediction)
        
        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            x, state = self.lstm_cell_decoder(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction1 = self.dense1(x)
            prediction = self.dense2(prediction1)
            # Add the prediction to the output
            predictions.append(prediction)
        
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


def GetResult_inverseTransfrom(x_target, y_target, scaler, model_created, target_size):
    target_inverse_prepare = y_target.reshape(-1, 1)
    target_inversed = scaler.inverse_transform(target_inverse_prepare).reshape(-1, target_size, 1)
    target_inversed = target_inversed.sum(axis=1)

    pred = model_created.predict(x_target)
    pred_inverse_prepare = pred.reshape(-1, 1)
    pred_inversed = scaler.inverse_transform(pred_inverse_prepare).reshape(-1, target_size, 1)
    pred_inversed = pred_inversed.sum(axis=1)

    print(mean_absolute_error(target_inversed.flatten(), pred_inversed.flatten()))

    target_graph = target_inversed.flatten()
    pred_graph = pred_inversed.flatten()

    y_min = min(min(target_graph), min(pred_graph))-100
    y_max = max(max(target_graph), max(pred_graph))+100

    # y_min = min(min(target_inversed.flatten()), min(pred_inversed.flatten()))-100
    # y_max = max(max(target_inversed.flatten()), max(pred_inversed.flatten()))+100

    plt.rcParams["figure.figsize"] = (16, 5)
    # plt.plot(target_inversed.flatten(), 'r.-', label='predict')
    # plt.axis([1, len(target_inversed.flatten()), y_min, y_max])
    plt.plot(pred_graph, 'r.-', label='predict')
    plt.axis([1, len(pred_graph), y_min, y_max])
    plt.legend(fontsize=14)

    # plt.plot(pred_inversed.flatten(), 'b.-', label='original')
    # plt.axis([1, len(pred_inversed.flatten()), y_min, y_max])
    plt.plot(target_graph, 'b.-', label='original')
    plt.axis([1, len(target_graph), y_min, y_max])
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

    return mean_absolute_error(target_inversed.flatten(), pred_inversed.flatten())
    # print(mean_absolute_error(target_inversed.sum(axis=1), pred_inversed.sum(axis=1)))


def plot_learning_curves(loss, val_loss, epochs):
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 0.5, loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_learning_curves_train(loss, epochs):
    plt.rcParams["figure.figsize"] = (7, 6)
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    # plt.plot(np.arange(len(val_loss)) + 0.5, loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

my_prediction = {}

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
          ]

def Graph_Evaluation(mae, name='no_name'):
    global my_prediction

    my_prediction[name] = mae
    y_value = sorted(my_prediction.items(), key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(y_value, columns=['algorithm', 'mae'])
    min_ = df['mae'].min() - 10
    max_ = df['mae'].max() + 10

    length = len(df)

    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['algorithm'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mae'])

    for i, v in enumerate(df['mae']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    plt.title('Mean Absolute Error', fontsize=18)
    plt.xlim(min_, max_)

    plt.show()


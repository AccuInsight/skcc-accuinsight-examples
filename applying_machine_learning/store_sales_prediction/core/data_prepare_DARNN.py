import numpy as np
import pandas as pd


class Preprocess_DARNN():
    def __init__(self, train_data, test_data, interval):
        self.Y_train = train_data.iloc[:, 0]
        self.X_train = train_data.iloc[:, 1:]
        self.Y_test = test_data.iloc[:, 0]
        self.X_test = test_data.iloc[:, 1:]
        self.interval = interval
        self.encoder_sequence_tr, self.decoder_sequence_tr, self.target_tr = \
            self.Create_dataset(self.X_train, self.Y_train)
        self.encoder_sequence_ts, self.decoder_sequence_ts, self.target_ts = \
            self.Create_dataset(self.X_test, self.Y_test)


    def Create_dataset(self, X_df, Y_df):
        encoder_list = []
        decoder_list = []
        target_list = []

        for i in range(1, X_df.shape[0] - self.interval):
            encoder_list.append(np.array(X_df.iloc[i: i + self.interval, :]))
            decoder_list.append(np.array(Y_df.iloc[i: i + self.interval - 1]))
            target_list.append(Y_df.iloc[i + self.interval - 1])

        encoder_sequence = np.array(encoder_list)
        decoder_sequence = np.array(decoder_list)

        decoder_sequence = np.reshape(decoder_sequence, (-1, self.interval - 1, 1))
        target = np.array(target_list)

        return encoder_sequence, decoder_sequence, target

    def Show_shape(self, option):
        if option == 'train':
            print('Shape of encoder input : {}'.format(self.encoder_sequence_tr.shape))
            print('Shape of decoder input : {}'.format(self.decoder_sequence_tr.shape))
            print('Shape of target input : {}'.format(self.target_tr.shape))
        else:
            print('Shape of encoder input : {}'.format(self.encoder_sequence_ts.shape))
            print('Shape of decoder input : {}'.format(self.decoder_sequence_ts.shape))
            print('Shape of target input : {}'.format(self.target_ts.shape))
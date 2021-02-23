import numpy as np
import tensorflow as tf
import sys
from common import *
# PMD_DIR = os.path.join(ROOT_DIR, 'store_pmd')
# SOURCE_DIR = os.path.join(PMD_DIR, 'source')

# sys.path.append(SOURCE_DIR)
from display_stand.core.Dual_stage_attention_model import DARNN
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from display_stand.core.data_prepare_DARNN import *


class model_DARNN():
    def __init__(self, train_data, test_data, interval, m, p, n, batch_size, learning_rate, epochs):
        '''
        :param interval : interval of time series
        :param m: encoder lstm unit length
        :param p: decoder lstm unit length
        :param n: number of features
        '''
        self.pre_darnn = Preprocess_DARNN(train_data, test_data, interval)
        self.pre_darnn.Show_shape(option='train')
        print('---------------------------------------')
        self.pre_darnn.Show_shape(option='test')

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = DARNN(T=interval, m=m, p=p, n=n)
        self.interval = interval
        self.n = n
        self.train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.pre_darnn.encoder_sequence_tr, self.pre_darnn.decoder_sequence_tr, self.pre_darnn.target_tr)
            )
            .batch(self.batch_size)
            .shuffle(buffer_size=self.pre_darnn.encoder_sequence_tr.shape[0])
            .prefetch(tf.data.experimental.AUTOTUNE)
            )

        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.pre_darnn.encoder_sequence_ts, self.pre_darnn.decoder_sequence_ts, self.pre_darnn.target_ts)
            )\
            .batch(self.batch_size)

    def createModel(self):
        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.pre_darnn.encoder_sequence_tr, self.pre_darnn.decoder_sequence_tr, self.pre_darnn.target_tr)
            )
                .batch(self.batch_size)
                .shuffle(buffer_size=self.pre_darnn.encoder_sequence_tr.shape[0])
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (self.pre_darnn.encoder_sequence_ts, self.pre_darnn.decoder_sequence_ts, self.pre_darnn.target_ts)
        ) \
            .batch(self.batch_size)

        @tf.function
        def train_step(model, inputs, labels, loss_fn, optimizer, train_loss):
            with tf.GradientTape() as tape:
                prediction = model(inputs, training=True)
                loss = loss_fn(labels, prediction)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        @tf.function
        def test_step(model, inputs, labels, loss_fn, test_loss):
            prediction = model(inputs, training=False)
            loss = loss_fn(labels, prediction)
            test_loss(loss)
            return prediction

        loss_fn = tf.keras.losses.MSE

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        test_loss = tf.keras.metrics.Mean(name="test_loss")
        train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
        test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")
        history_loss = []

        for epoch in range(self.epochs):
            for enc_data, dec_data, labels in train_ds:
                inputs = [enc_data, dec_data]
                train_step(self.model, inputs, labels, loss_fn, optimizer, train_loss)

            template = "Epoch {}, Loss: {}"
            print(template.format(epoch + 1, train_loss.result()))
            history_loss.append(train_loss.result().numpy())
            train_loss.reset_states()
            test_loss.reset_states()

        i = 0
        for enc_data, dec_data, label in test_ds:
            inputs = [enc_data, dec_data]
            pred = test_step(self.model, inputs, label, loss_fn, test_loss)
            if i == 0:
                preds = pred.numpy()
                labels = label.numpy()
                i += 1
            else:
                preds = np.concatenate([preds, pred.numpy()], axis=0)
                labels = np.concatenate([labels, label.numpy()], axis=0)
        print(test_loss.result(), test_accuracy.result() * 100)

        return preds, labels, history_loss


    def coeff_InputAttention(self, variable_dict):

#         font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#         rc("font", family=font_name)
        variable_key = list(variable_dict.keys())
        alpha = []
        variables = []
        for i in range(self.n):
            alpha.append(np.mean(self.model.encoder.alpha_t[:, 0, i].numpy()))
            for key in variable_key:
                if f"{i}" in variable_dict[key]:
                    variables.append(f"{key}{i}")
        plt.figure(figsize=(6, 4))
        plt.bar(x=variables, height=alpha, color="navy")
        plt.style.use("seaborn-pastel")
        plt.title("alpha")
        plt.xlabel("variables")
        plt.xticks(rotation=90)
        plt.ylabel("prob")
        plt.show()


    def coeff_TemporalAttention(self):
        enc_data, dec_data, label = next(iter(self.test_ds))
        inputs = [enc_data, dec_data]

        pred = self.model(inputs)
        beta = []
        for i in range(self.interval-1):
            beta.append(np.mean(self.model.decoder.beta_t[:, i, 0].numpy()))
        plt.bar(x=range(self.interval-1), height=beta, color="navy")
        plt.style.use("seaborn-pastel")
        plt.title("Beta")
        plt.xlabel("time")
        plt.ylabel("prob")
        plt.show()

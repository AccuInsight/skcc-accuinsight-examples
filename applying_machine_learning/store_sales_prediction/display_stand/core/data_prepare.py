
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Subject: Data Preparation for PMD
    Copyright (C) 2019 of Data Science TF. All rights reserved.
    Licence: SK Holdings C&C, Data Science TF
    Status: Development
    Version: 0.9
    Python Version: 3.7.x
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
pd.set_option('display.max_columns', None)


class SalesData():
    """
    :parameter
        1. data_all : preprocessed data (ready to run)
        2. split_time : time of splitting train/test set
        3. target_store: target store
        4. target_product: target product

    :Process
        read pre-processed file
        fill missing days and order
        apply min max scale
        make data set and split to train and test
    """

    def __init__(
            self,
            data_all,
            time_size,
            target_size,
            # batch_size,
            # num_epochs,
            split_time,
            target_store,
            target_product
    ):

        self.data_all = data_all
        self.target_store = target_store
        self.target_product = target_product
        self.split_time = split_time
        self.time_size = time_size
        self.target_size = target_size
        # self.batch_size = batch_size
        # self.num_epochs = num_epochs

        self.df_train, self.df_test, self.df_training_date, self.df_test_date = self.filter_data_all()

        self.train_df, self.train_scl_y, \
            self.train_scl_p, self.train_scl_i = self.prepare_scale(option="train")

        self.test_df, self.test_scl_y, \
            self.test_scl_p, self.test_scl_i = self.prepare_scale(option="test")

        self.trainSet = self.make_dataset(option="train")
        self.testSet = self.make_dataset(option="test")

        self.x_trainFinal, self.y_trainFinal = self.getTrainBatch()
        self.x_testFinal, self.y_testFinal = self.getTestBatch()

    def filter_data_all(self):

        data_all = self.data_all
        target_store = self.target_store
        target_product = self.target_product

        data_target = data_all.loc[(data_all.store == target_store)
                                   & (data_all.product_c == target_product)]
        data_target['weekend'] = 0.0

        df = pd.DataFrame({'date': pd.date_range(min(data_target.date), end=max(data_target.date))})
        df['date'] = pd.to_datetime(df['date']).dt.date
        data_target.date = pd.to_datetime(data_target['date']).dt.date
        merged_sales = pd.merge(df, data_target, how='left', on='date')
        merged_sales['date'] = pd.to_datetime(merged_sales['date'], format='%Y-%m-%d', errors='coerce')

        merged_sales["store"] = target_store
        merged_sales["product_c"] = target_product
        merged_sales = merged_sales.fillna(0)

        merged_sales['week'] = pd.to_datetime(merged_sales['date']).dt.dayofweek
        merged_sales.loc[merged_sales['week'] > 4, 'weekend'] = 1.0

        df_training = merged_sales.loc[merged_sales['date'] < self.split_time]
        df_test = merged_sales.loc[merged_sales['date'] >= self.split_time]

        df_training_date = df_training
        df_test_date = df_test

        df_training = df_training.drop(columns=['date', 'store', 'product_c', 'week'])
        df_test = df_test.drop(columns=['date', 'store', 'product_c', 'week'])

        return df_training, df_test, df_training_date, df_test_date

    def prepare_scale(self, option):

        if option == "train":
            merged_sales = self.df_train
        else:
            merged_sales = self.df_test

        def scale_and_transform(data):
            data_ = np.array(data)
            data_scaler = MinMaxScaler(feature_range=(-1, 1))
            data_scaler.fit(data_.reshape((-1, 1)))
            data_ = data_scaler.transform(data_.reshape((-1, 1)))
            data_ = data_.reshape(-1)
            return data_, data_scaler

        y_, data_scaler_y = scale_and_transform(merged_sales.sales)
        p_, data_scaler_p = scale_and_transform(merged_sales.pmd_ratio)
        i_, data_scaler_i = scale_and_transform(merged_sales.weekend)

        li = [y_,p_,i_]
        sales_data = pd.DataFrame(data=li).transpose()
        sales_data.columns = list(merged_sales)

        return sales_data, data_scaler_y, data_scaler_p, data_scaler_i


    def make_dataset(self, option):
        """
        :argument
        if train, make dataset as train_df from scaled dataset
        :return
        train and test data set
        """

        if option == "train":
            sales_data = self.train_df
        else:
            sales_data = self.test_df

        maxday = self.time_size  # 14   # x 값으로 볼 time series 길이
        maxtarget = self.target_size  # 7  # y 값으로 볼 time series 길이
        dataset = []

        for i in range(maxday, len(sales_data) - maxtarget+1):
            print('making dataset progress : {}/{}'.format(i, len(sales_data)), end='\r')

            historySet = sales_data.sales.loc[i - self.time_size:i - 1]
            pmdSet = sales_data.pmd_ratio.loc[i - self.time_size:i - 1]
            weekendSet = sales_data.weekend.loc[i - self.time_size:i - 1]
            targetSet = sales_data.sales.loc[i:i + self.target_size - 1]

            target_history = np.reshape(np.array(historySet), (self.time_size, 1))
            pmd_history = np.reshape(np.array(pmdSet), (self.time_size, 1))
            weekend_history = np.reshape(np.array(weekendSet), (self.time_size, 1))
            target_sales = np.reshape(np.array(targetSet), (self.target_size, 1))

            dataset.append(
                {'target_history': target_history,
                 'pmd_history': pmd_history,
                 'weekend_history': weekend_history,
                 'target_sales': target_sales,
                 }
                )

        return dataset


    def getTrainBatch(self):

        """
        divide sets to y,xb,xm,xi,target and create train and eval batch.
        args:
            option='training' or None('evaluation')
        returns:
            batch={'y','xp','xi','target'}
        """
        returnSet = self.trainSet

        y = []
        xp = []
        xi = []
        target = []

        for d in returnSet:
            y.append(d['target_history'])
            xp.append(d['pmd_history'])
            xi.append(d['weekend_history'])
            target.append(d['target_sales'])

        y = np.reshape(y, (-1, self.time_size, 1))  # (# of batch_size(window에 따라 데이터 뽑았을 때 데이터 갯수), time_size, # of feature(=1))
        xp = np.reshape(xp, (-1, self.time_size, 1))
        xi = np.reshape(xi, (-1, self.time_size, 1))
        target = np.reshape(target, (-1, self.target_size, 1))

        x_trainFinal = np.reshape(np.stack((y, xp, xi), axis=2), (-1, self.time_size, 3))
        y_trainFinal = target

        return x_trainFinal, target



    def getTestBatch(self):
        """
        split dataset saved in class into y,x_b,x_m,x_w,x_s,target and create batches
        :returns
            batch={'y','xb','xm','xw','xs','target'}
        """
        returnSet = self.testSet

        y = []
        xp = []
        xi = []
        target = []

        for d in returnSet:
            y.append(d['target_history'])
            xp.append(d['pmd_history'])
            xi.append(d['weekend_history'])
            target.append(d['target_sales'])

        y = np.reshape(y, (-1, self.time_size, 1))  # (# of batch_size(window에 따라 데이터 뽑았을 때 데이터 갯수), time_size, # of feature(=1))
        xp = np.reshape(xp, (-1, self.time_size, 1))
        xi = np.reshape(xi, (-1, self.time_size, 1))
        target = np.reshape(target, (-1, self.target_size, 1))

        x_testFinal = np.reshape(np.stack((y, xp, xi), axis=2), (-1, self.time_size, 3))
        y_testFinal = target

        return x_testFinal, y_testFinal


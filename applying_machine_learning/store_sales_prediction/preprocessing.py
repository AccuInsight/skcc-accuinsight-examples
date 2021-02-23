#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Subject: Preprocessing
    
    Copyright (C) 2019 of Data Science TF. All rights reserved.
    Licence: SK Holdings C&C, Data Science TF
    
    Status: Development
    Version: 0.9
    
    Python Version: 3.6.8
"""
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import os
import datetime
pd.set_option('display.max_columns', None)


class ProcessSale():

    """
    ;parameter
        data_path : path where data is stored
        processed_path : path where processed data will be stored
        store_type: [store_offline, online, store_pmd, dc_offline]
        input_file : original sales data
        input_eos : information of end of sales products

    ;process
        1. Load data (plz check the required input format below)
        2. Preprocess data
        3. Filter data (final process)
    """

    def __init__(self, data_path, processed_path, store_type, input_file, input_eos):

        self.path = data_path
        self.processed_path = processed_path
        self.store_type = store_type
        self.input_file = input_file
        self.input_eos = input_eos

        self.sales_original = self.load_data()
        self.sales_preprocessed = self.preprocess_data()

    def load_data(self):
        """
        ;file:
        (offline)
            1. Required format: '.csv'
            2. Required columns (in order): 기준일자,매장코드,상품코드,판매수량
        (others)
            1. Required format: '.csv'
            2. Required columns (in order): 기준일자,상품코드,판매수량(온라인: 주문수량)

        ;column format details:
            1. 기준일자: should be '%Y-%m-%d'
            2. 판매수량: should be integer/number (Plz be aware that these columns sometimes does include
            'comma', which cause value errors)

        :return:
            Initial sales data, only column names and date format replaced
        """
        path = self.path
        input_file = self.input_file

        if self.store_type == 'store_offline':
            df = pd.read_csv(os.path.join(path, input_file))
            df.columns = ['date', 'store', 'product_c', 'sales']
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        elif self.store_type == 'online':
            df = pd.read_csv(os.path.join(path, input_file))
            df.columns = ['date', 'product_c', 'sales']
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df['store'] = 'S023'
        else:
            df = pd.read_csv(os.path.join(path, input_file))
            df.columns = ['date', 'product_c', 'sales']
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df['store'] = 'all'

        return df

    def load_eos(self):
        """
        ;file
            1. Required format: '.csv'
            2. Required columns: 상품코드,상품상태명
        :return:
            product code of End of Sales
        """
        path = self.path
        prod_eos = self.input_eos

        products_eos = pd.read_csv(os.path.join(path, prod_eos))
        products_eos.columns = ['product_c', 'status']
        prod_eos = products_eos.loc[products_eos.status.isin(['발주종료', '판매종료']), 'product_c'].unique().tolist()

        return prod_eos

    def preprocess_data(self):
        """
        ;process
            1. Replace negative sales with 0
            2. Group by (date and product_c) and sum 'sales' so that a product per day can contain only one value
        :return:
            preprocessed data
        """
        df = self.sales_original

        df['sales'] = np.where(df['sales'] < 0, 0.0, df['sales'])
        df = df.groupby(['date', 'store', 'product_c']).agg({'sales': 'sum'}).reset_index()
        df = df.loc[:, ['date', 'store', 'product_c', 'sales']]

        return df

    def final_process(self):
        """
        ;process
            1. Filter data for no sales for the last 6 months
            2. Keep only the products w/ (at least) once a week average sales
            3. Remove products no longer available
        :return:
            final, filtered data
        """
        sales = self.sales_preprocessed
        prod_eos = self.load_eos()
        store_type = self.store_type

        # 1. Zero sales for the last 6 Months
        sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d', errors='coerce')
        target_day = sales.date.max() + relativedelta(months=-6)
        if (sales.date.nunique()) == (sales.date.max() - sales.date.min() + relativedelta(days=1)):
            print("You may have errors in sales.date data!!")

        check_product = sales[sales.date >= target_day]
        check_product = check_product.groupby(['store', 'product_c']).agg({'sales': 'sum'}).reset_index()
        check_product = check_product.loc[check_product.sales != 0, ['product_c']]
        new_sales = pd.merge(check_product, sales, how='inner', on=['product_c']).reset_index()
        new_sales = new_sales.loc[:, ['date', 'store', 'product_c', 'sales']]

        # 2. Rotation by week (at least one sales per week, which equals to 0.142857142)
        w_mean = new_sales.groupby(['store', 'product_c']).agg({'sales': 'mean'}).reset_index()
        w_mean2 = w_mean[w_mean.sales >= 0.142857142]
        check_rotation = w_mean2.product_c.unique().tolist()

        new_sales2 = new_sales.loc[new_sales.product_c.isin(check_rotation)]

        # 3. Remove End of Sales products
        sales_final = new_sales2.loc[~new_sales2.product_c.isin(prod_eos), :]

        if (store_type == 'online') or (store_type == 'dc_offline'):
            time_str = str(int(datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
            file_name = store_type + '_' + time_str + '.csv'
            sales_final.to_csv(os.path.join(self.processed_path, file_name), index=False)

        return sales_final


class ProcessSalePromotion(ProcessSale):

    """
    This class is inherited from ProcessSale class for those data needed to be merged w/ promotion data

    :parameter
        1. promotions: list of promotion type ([brand, ppm, scp] or [pmd, scp])
    """

    def __init__(self, data_path, processed_path, store_type, input_file, input_eos, promotions):

        ProcessSale.__init__(self, data_path, processed_path, store_type, input_file, input_eos)
        self.promotions = promotions
        self.promotion_merged = self.load_promotion()
        self.sale_final_ready = self.final_process()

    def load_promotion(self):
        """
        ;process:
            1. Bring promotion information
            2. Change the column names in Eng and date format
            3. Filter sale_type to remove '증정' etc...
            4. Adjust ratio to be pure numeric
            5. Spread date
            6. Remove dups
            7. Finally, merge all the sales in promotions list ([brand, ppm, scp] or [pmd, scp])

        ;file
            1. Required format" '.csv'
            2. Required file name: '~_sale_info.csv' e.g. 'brand_sale_info.csv', 'pmd_sale_info.csv'
            3. Required columns: (소)행사시작일,(소)행사종료일,상품코드,오퍼구분명,매가DC율(오퍼율)

        ;column format details:
            1. (소)행사시작일 & (소)행사종료일: should be '%Y%m%d'

        :return:
            sale information
        """
        path = self.path
        promotions = self.promotions

        sale_all = pd.DataFrame(columns=['date', 'product_c'])

        for sale_type in promotions:
            sale = pd.read_csv(path + '/' + sale_type + '_sale_info.csv')
            sale.columns = ['start_date', 'end_date', 'product_c', 'type', 'ratio']
            sale['start_date'] = pd.to_datetime(sale['start_date'], format='%Y%m%d', errors='coerce')
            sale['end_date'] = pd.to_datetime(sale['end_date'], format='%Y%m%d', errors='coerce')
            sale = sale.loc[~sale.product_c.isna(), :]
            sale['product_c'] = sale['product_c'].astype(int)

            # 1. Remove '증정' & '적립' & '엔드' just in case (In case of PMD, we only use '엔드')
            if sale_type == 'pmd':
                sale = sale[(sale['type'].str.contains('증정') == False) & (sale['type'].str.contains('적립') == False)]
            else:
                sale = sale[(sale['type'].str.contains('증정') == False) & (sale['type'].str.contains('적립') == False) & (
                        sale['type'].str.contains('엔드') == False)]

            # 2. Adjust 'ratio'
            rm_ratio = ['없음', 'A+B(FreeGift)']
            bundle1_ratio = ['1+1(단일)', '1+1(교차)']
            bundle2_ratio = ['2+1(단일)', '2+1(교차)']

            sale = sale[~sale['ratio'].isin(rm_ratio)]
            sale['ratio'] = np.where(sale['ratio'].isin(bundle1_ratio), 45.0, sale['ratio'])
            sale['ratio'] = np.where(sale['ratio'].isin(bundle2_ratio), 29.7, sale['ratio'])
            sale['ratio'] = sale['ratio'].apply(float).fillna(0.0)
            sale['ratio'] = np.where(sale['ratio'] <= 1.0, sale['ratio'] * 100, sale['ratio'])
            sale['ratio'] = np.where(sale['ratio'] < 0.0, 0.0, sale['ratio'])
            sale = sale.loc[sale['ratio'] >= 0, :]

            # 3. Spread start date and end date of promotion
            sale_df = pd.concat([pd.DataFrame(
                {'date': pd.date_range(row.start_date, row.end_date), 'product_c': row.product_c, 'ratio': row.ratio},
                columns=['date', 'product_c', 'ratio']) for i, row in sale.iterrows()], ignore_index=True)

            # 4. To remove duplicated promotion ratio on the same date
            sale_df = sale_df.drop_duplicates()
            sale_df = sale_df.groupby(['date', 'product_c']).agg({'ratio': 'sum'}).reset_index()

            sale_df.columns = ['date', 'product_c', str(sale_type) + '_ratio']

            # 5. Merge all the required sale types
            sale_all = sale_all.merge(sale_df, on=['date', 'product_c'], how='outer')

        return sale_all

    def final_process_promotion(self):
        """

        :return: final sales data w/ required promotion information
        """
        sales = self.sale_final_ready
        promotion = self.promotion_merged

        if self.store_type == 'store_offline':
            sales = sales.merge(promotion, on=['date', 'product_c'], how='left')
            sales['brand_ratio'] = sales['brand_ratio'].fillna(0.0)
            sales['ppm_ratio'] = sales['ppm_ratio'].fillna(0.0)
            sales['scp_ratio'] = sales['scp_ratio'].fillna(0.0)
            sales['sales'] = np.where(sales.scp_ratio > 0, 0.0, sales['sales'])
            sales = sales.drop(columns=['scp_ratio'])

        elif self.store_type == 'store_pmd':
            sales = sales.merge(promotion, on=['date', 'product_c'], how='left')
            sales['pmd_ratio'] = sales['pmd_ratio'].fillna(0.0)
            sales['scp_ratio'] = sales['scp_ratio'].fillna(0.0)
            sales['sales'] = np.where(sales.scp_ratio > 0, 0.0, sales['sales'])
            sales = sales.drop(columns=['scp_ratio'])

        sales_final = sales.drop_duplicates()

        time_str = str(int(datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
        file_name = self.store_type + '_' + time_str + '.csv'
        sales_final.to_csv(os.path.join(self.processed_path, file_name), index=False)

        return sales_final



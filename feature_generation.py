#!/usr/bin/env/ python2
# -*- coding:utf-8 -*-
from __future__ import (
    print_function,
    division,
)
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as u
import uncertainties.unumpy as unp

from numpy.core.defchararray import count

matplotlib.style.use('ggplot')

NUMS = (1, 2, 3)
PLOT = True

def feature_generation(df, brands=None, categories=None):
    df = df.copy()

    df = add_timing(df)
    df = add_log10(df, 'deltaT')

    if brands is not None:
        df = add_brands(df, brands)

    if categories is not None:
        df = add_categories(df, categories)

    df = add_periodical_transform(df, 'orderTime_weekday', 7)
    df = add_periodical_transform(df, 'orderTime_minutes', 1440)
    df = add_periodical_transform(df, 'couponsReceived_weekday', 7)
    df = add_periodical_transform(df, 'couponsReceived_minutes', 1440)

    for num in NUMS:
        df = add_log10(df, 'price{}'.format(num))
        df = add_log10(df, 'basePrice{}'.format(num))

    df['priceSum'] = df.price1 + df.price2 + df.price3
    df = add_log10(df, 'priceSum')

    return df


def add_brands(df, brands):
    """
    adds binary features for the product brands to df
    """
    df = df.copy()
    brand_cols = ['brand1', 'brand2', 'brand3']
    for b in brand_cols:
        df[b].astype('category').cat.set_categories(brands)
        df = df.join(pd.get_dummies(df[b], b, dummy_na=True))

    return df


def get_all_brands(df1, df2):
    """
    get all brands from both datasets
    """
    brand_cols = ['brand1', 'brand2', 'brand3']
    brands1 = set(df1[brand_cols].values.flatten())
    brands2 = set(df2[brand_cols].values.flatten())

    return brands1.union(brands2)


def add_log10(df, feature):
    """
    adds log10 of feature as logFeature
    """
    df = df.copy()
    df['log' + feature[0].capitalize() + feature[1:]] = np.log10(df[feature])
    return df


def add_timing(df):
    """
    adds some timing features
    """

    df = df.copy()

    df.orderTime = pd.to_datetime(df.orderTime)
    df.couponsReceived = pd.to_datetime(df.couponsReceived)
    df['deltaT'] = (df.orderTime - df.couponsReceived).astype('int64')
    df['orderTime_weekday'] = df.orderTime.dt.dayofweek
    df['couponsReceived_weekday'] = df.couponsReceived.dt.dayofweek
    df['orderTime_minutes'] = df.orderTime.dt.hour * 60 + df.orderTime.dt.minute
    df['couponsReceived_minutes'] = (df.couponsReceived.dt.hour * 60 +
                                     df.couponsReceived.dt.minute)
    df['sameDay'] = df.orderTime.dt.dayofyear == df.couponsReceived.dt.dayofyear

    return df


def add_categories(df, categories):
    """
    adds a lot of binary features for the categories
    """
    df = df.copy()
    for num in NUMS:
        for i, cat in enumerate(categories):
            category_strings = df["categoryIDs{}".format(num)].values.astype(str)
            category_found = count(category_strings, cat) > 0
            df["product{}_cat{}".format(num, i)] = category_found
    return df


def get_all_categories(df1, df2):
    """
    finds the unique categories in both datasets
    """
    category_cols = ['categoryIDs1', 'categoryIDs2', 'categoryIDs3']
    categoriesString = ",".join(
        np.hstack([
            df1[category_cols].values.flatten(),
            df2[category_cols].values.flatten(),
        ])
    )
    categories = set(categoriesString.split(","))

    return categories


def add_periodical_transform(df, feature, period):
    df = df.copy()

    transform = np.sin(df[feature] * 2 * np.pi/period)
    df['periodic' + feature[0].capitalize() + feature[1:]] = transform

    return df


if __name__ == '__main__':

    train = pd.read_csv('data/train.txt', delimiter='|')
    classify = pd.read_csv('data/class.txt', delimiter='|')

    n_before = len(train.columns)

    categories = get_all_categories(train, classify)
    brands = get_all_brands(train, classify)

    train = feature_generation(train, brands, categories)
    classify = feature_generation(classify, brands, categories)

    print('created {} new feautes'.format(len(train.columns) - n_before))

    train.to_csv("build/train_with_new_features.txt", sep="|")
    classify.to_csv("build/class_with_new_features.txt", sep="|")



    #
    # Control plots
    #
    # Correlation
    if PLOT:
        corr = train.corr()
        plt.figure(figsize=(25, 25))
        plt.imshow(corr, cmap='seismic', vmin=-1, vmax=1, interpolation='none')
        plt.colorbar()
        plt.xticks(np.arange(0, len(corr.columns)), corr.columns,
                   rotation='vertical')
        plt.yticks(np.arange(0, len(corr.columns)), corr.columns)
        plt.grid('off')
        plt.tight_layout()
        plt.savefig('plots/correlation.pdf', bbox_inches='tight')
        plt.clf()
    # Separation
        variables = ('deltaT', 'logDeltaT', 'orderTime_weekday',
                     'orderTime_minutes', 'sameDay', 'premiumProduct{}', 'price{}',
                     'basePrice{}', 'reward{}', 'logPrice{}', 'logBasePrice{}')
        ncols = 3
        nrows = len(variables)
        plt.figure(figsize=(ncols * 5, nrows * 4))
        for num in NUMS:
            crit = 'coupon{}Used'.format(num)
            sig = train[train[crit] == 1]
            bkg = train[train[crit] == 0]
            for i, var in enumerate(variables):
                if '{}' in var:
                    var = var.format(num)
                plt.subplot(nrows, ncols, i * ncols + num)
                _, bins, _ = plt.hist(sig[var].values, alpha=0.5, normed=True)
                plt.hist(bkg[var].values, bins=bins, alpha=0.5, normed=True)
                plt.xlabel('{}'.format(var), ha='right', x=1)
                plt.ylabel('relative frequency', ha='right', y=1)
        plt.tight_layout()
        plt.savefig('plots/separation.pdf', bbox_layout='tight')
        plt.clf()

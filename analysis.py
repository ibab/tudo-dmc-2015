#!/usr/bin/env/ python2
# -*- coding:utf-8 -*-
from __future__ import (
    print_function,
    division,
    unicode_literals
)

import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as u
import uncertainties.unumpy as unp

import sklearn as skl
from sklearn.cross_validation import (
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor


def score2ufloat(score):
    return u.ufloat(score.mean(), score.std())


def prudsys_score(estimator, X, y):
    prediction = estimator.predict(X)
    score = np.sum(((prediction - y)/np.mean(y, axis=0))**2)

    return score


df = pd.read_csv("build/train_with_new_features.txt", sep=b"|")

columns = [
    'deltaT',
    'logDeltaT',
    'price1',
    'price2',
    'price3',
    'basePrice1',
    'basePrice2',
    'basePrice3',
    'reward1',
    'reward2',
    'reward3',
    'premiumProduct1',
    'premiumProduct2',
    'premiumProduct3',
    'orderTime_minutes',
    'orderTime_weekday',
    'couponsReceived_minutes',
    'couponsReceived_weekday',
    'sameDay',
    'priceSum',
]

# columns.extend([b for b in df.columns
#                 if 'brand' in b and b not in ('brand1', 'brand2', 'brand3')])

# columns.extend([cat for cat in df.columns
#                 if 'cat' in cat and cat not in ("categoryIDs1",
#                                                 "categoryIDs2",
#                                                 "categoryIDs3")])

labels = [
    'coupon1Used',
    'coupon2Used',
    'coupon3Used',
    'basketValue',
]

features = df[columns].values
labels = df[labels].values
learners = {
    'Random Forest': RandomForestRegressor(
        n_estimators=250,
        n_jobs=-1,
        max_features="auto",
    ),
    'Decision Tree': DecisionTreeRegressor()
}




x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
for name, learner in learners.iteritems():
    score = cross_val_score(learner, features, labels, scoring=prudsys_score, n_jobs=-1)

    learner.fit(x_train, y_train)
    rek_basket = learner.predict(x_test)

    plt.hist2d(np.log10(rek_basket[:,3]), np.log10(y_test[:,3]), 100, cmap="hot")
    plt.xlabel("log10(estimated basketValue)")
    plt.ylabel("log10(true basketValue)")
    plt.colorbar()
    plt.savefig('plot_{}.pdf'.format(name))
    plt.clf()

    result = score2ufloat(score)
    print("{}: {:1.5f}+/-{:1.5f}".format(name, result.n, result.s))

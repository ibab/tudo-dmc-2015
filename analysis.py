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
    KFold,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from sklearn.svm import (
    SVC,
    SVR,
)
from sklearn.tree import DecisionTreeRegressor


def score2ufloat(score):
    return u.ufloat(score.mean(), score.std())


def prudsys_score_xval(estimator, X, y):
    prediction = estimator.predict(X)
    score = np.sum(((prediction - y)/np.mean(y, axis=0))**2)/len(y)
    return score

def prudsys_score(prediction, truth):
    score = np.sum(((prediction - truth)/np.mean(truth, axis=0))**2)/len(truth)
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
    'periodicOrderTime_minutes',
    'periodicOrderTime_weekday',
    'couponsReceived_minutes',
    'couponsReceived_weekday',
    'periodicCouponsReceived_minutes',
    'periodicCouponsReceived_weekday',
    'premiumProduct1',
    'premiumProduct2',
    'premiumProduct3',
    'sameDay',
    'priceSum',
    'reward1',
    'reward2',
    'reward3',
]

# columns.extend([b for b in df.columns
#                 if 'brand' in b and b not in ('brand1', 'brand2', 'brand3')])

# columns.extend([cat for cat in df.columns
#                 if 'cat' in cat and cat not in ("categoryIDs1",
#                                                 "categoryIDs2",
#                                                 "categoryIDs3")])

labelnames_classification = [
    'coupon1Used',
    'coupon2Used',
    'coupon3Used',
]

labelnames_regression = [
    'basketValue',
]

features = df[columns].values
labels_classification = df[labelnames_classification].values
labels_regression = df[labelnames_regression].values

classifier = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
)

# regressor = RandomForestRegressor(
#     n_estimators=200,
#     n_jobs=-1,
# )

regressor = SVR()


scores = []

kfv = KFold(features.shape[0], n_folds=10, shuffle=True)
for i, (train, test) in enumerate(kfv):

    print('cross val run {}'.format(i+1))

    # predict the couponsXUsed variables:
    classifier.fit(features[train], labels_classification[train])
    coupon_pred = classifier.predict(features[test])

    print('Mean coupon1Used pred:', coupon_pred[:,0].mean())
    print('Mean coupon1Used truth:', labels_classification[test][:,0].mean())


    regressor.fit(features[train], labels_regression[train].ravel())
    basket_pred = regressor.predict(features[test])

    prediction = np.column_stack([coupon_pred, basket_pred])
    truth = np.column_stack([labels_classification[test],
                             labels_regression[test],
                             ])

    scores.append(prudsys_score(prediction, truth))

print(u'Score: {:3.3f} ± {:3.3f}'.format(np.mean(scores), np.std(scores)))

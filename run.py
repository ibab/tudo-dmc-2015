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
import sklearn as skl
import uncertainties as u
import uncertainties.unumpy as unp

from numpy.core.defchararray import count

from sklearn.cross_validation import (
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor

def prudsys_score(estimator, X, y):
    prediction = estimator.predict(X)
    score = np.sum(((prediction - y)/np.mean(y, axis=0))**2)

    return score


matplotlib.style.use('ggplot')

NUMS = (1, 2, 3)
PLOT = False

#
# Load and prepare data
#
df = pd.read_csv('data/train.txt', delimiter='|')
# Prices
for num in NUMS:
    df['logPrice{}'.format(num)] = np.log(df['price{}'.format(num)])
    df['logBasePrice{}'.format(num)] = np.log(df['basePrice{}'.format(num)])

# Datetime
df.orderTime = pd.to_datetime(df.orderTime)
df.couponsReceived = pd.to_datetime(df.couponsReceived)
df['deltaT'] = (df.orderTime - df.couponsReceived).astype('int64')
df['logDeltaT'] = np.log(df.deltaT)
df['orderTime_weekday'] = df.orderTime.dt.dayofweek
df['couponsReceived_weekday'] = df.couponsReceived.dt.dayofweek
df['orderTime_minutes'] = df.orderTime.dt.hour * 60 + df.orderTime.dt.minute
df['couponsReceived_minutes'] = (df.couponsReceived.dt.hour * 60 +
                                 df.couponsReceived.dt.minute)
df['sameDay'] = df.orderTime.dt.dayofyear == df.couponsReceived.dt.dayofyear

df['priceSum'] = df['price1'] + df['price2'] + df['price3']

# Brands
brand_cols = ['brand1', 'brand2', 'brand3']
brands = set(df[brand_cols].values.flatten())
for b in brand_cols:
    df[b].astype('category').cat.set_categories(brands)
    df = df.join(pd.get_dummies(df[b], b, dummy_na=True))

# Categorys

categoriesString = ",".join(
    np.hstack([
        df.categoryIDs1,
        df.categoryIDs2,
        df.categoryIDs3,
    ])
)

categories = set(categoriesString.split(","))
for num in NUMS:
    for i, cat in enumerate(categories):
        category_strings = df["categoryIDs{}".format(num)].values.astype(str)
        category_found = count(category_strings, cat) > 0
        df["product{}_cat{}".format(num, i)] = category_found

#
# Control plots
#
# Correlation
if PLOT:
    corr = df.corr()
    plt.figure(figsize=(20, 20))
    plt.imshow(corr, cmap='seismic', vmin=-1, vmax=1, interpolation='none')
    plt.colorbar()
    plt.xticks(np.arange(0, len(corr.columns)), corr.columns,
               rotation='vertical')
    plt.yticks(np.arange(0, len(corr.columns)), corr.columns)
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
        sig = df[df[crit] == 1]
        bkg = df[df[crit] == 0]
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


#
# MVA
#
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

def score2ufloat(score):
    return u.ufloat(score.mean(), score.std())


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

    print("{}: {:P}".format(name, score2ufloat(score)))

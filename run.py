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

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor


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
# Brands
brands = reduce(
    lambda acc, x: acc.union(set(df['brand{}'.format(x)])),
    (1, 2, 3),
    set()
)
for num, brand in it.product(NUMS, brands):
    df['brand{}_{}'.format(num, brand)] = 0
for index, row in df.iterrows():
    df.loc[index, 'brand1_{}'.format(row.brand1)] = 1
    df.loc[index, 'brand2_{}'.format(row.brand2)] = 1
    df.loc[index, 'brand3_{}'.format(row.brand3)] = 1


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
    'sameDay'
]
columns.extend([b for b in df.columns
                if 'brand' in b and b not in ('brand1', 'brand2', 'brand3')])
labels = ['coupon1Used', 'coupon2Used', 'coupon3Used', 'basketValue']

X = df[columns].values
Y = df[labels].values
learners = {
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}


def score2ufloat(score):
    return u.ufloat(score.mean(), score.std())

for name, l in learners.iteritems():
    score = cross_val_score(l, X, Y, n_jobs=-1)
    print("{}: {:P}".format(name, score2ufloat(score)))

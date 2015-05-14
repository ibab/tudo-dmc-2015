from __future__ import absolute_import
from __future__ import print_function

from functools import partial
import numpy as np
import pandas as pd
from joblib import Memory
from joblib import Parallel, delayed
mem = Memory(cachedir='.joblib', verbose=0, mmap_mode='r')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(1337) # for reproducibility

def load_data(path):
    df = pd.read_csv(path, delimiter='|')
    data = df.copy()
    data.orderTime = pd.to_datetime(data.orderTime).astype('int64')/1e9/60
    data.couponsReceived = pd.to_datetime(data.couponsReceived).astype('int64')/1e9/60
    data['deltaT'] = (data.orderTime - data.couponsReceived)
    data['priceSum'] = data.price1 + data.price2 + data.price3
    data['meanCouponResponse1'] = -1

    meanResp = data.groupby('couponID1').mean()

    #for i, r in data.iterrows():
    #    print(i)
    #    used = data[r.orderID != data.orderID].groupby('couponID1').mean()['coupon1Used']
    #    if r.couponID1 in used.index:
    #        r['meanCouponResponse1'] = used.loc[r.couponID1]

    variables = ['orderTime',
                 'couponsReceived',
                 'deltaT',
                 'reward1',
                 'reward2',
                 'reward3',
                 'price1',
                 'price2',
                 'price3',
                 'priceSum',
                 'premiumProduct1',
                 'premiumProduct2',
                 'premiumProduct3',
                 'basePrice1',
                 'basePrice2',
                 'basePrice3',
                 #'meanCouponResponse1',
                 ]
    
    cat = ['brand1',
           'brand2',
           'brand3',
           'couponID1',
           'couponID2',
           'couponID3',
           'productGroup1',
           'productGroup2',
           'productGroup3',
           'categoryIDs1',
           'categoryIDs2',
           'categoryIDs3',
    ]

    X_var = data[variables].astype(np.float32).values

    x_cats = []
    for c in cat:
        xx = data[c].astype('str')
        encoder = LabelEncoder()
        encoder.fit(xx)
        x_cats.append(encoder.transform(xx))

    X_cat = np.vstack(x_cats).T
    X = np.hstack([X_var, X_cat])

    y = data[['coupon1Used', 'coupon2Used', 'coupon3Used']].values

    return X, y

def estimate_xgb(X_train, X_test, y_train, seed=0):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from rep.classifiers import XGBoostClassifier
    clf = XGBoostClassifier(n_estimators=35,
                            max_depth=4,
                            eta=0.35,
                            subsample=0.75,
                            gamma=18,
                            nthreads=2,
                            verbose=1,
                            random_state=seed)

    # Classify coupons one by one
    probs = []
    for i in [0, 1, 2]:
        proba = np.zeros(X_test.shape[0])
        clf.fit(X_train, y_train[:,i])
        proba = clf.predict_proba(X_test)[:,1]
        probs.append(proba)

    return np.array(probs).T

def estimate_xgb_20(X_train, X_test, y_train):
    proba = np.zeros((X_test.shape[0], y_train.shape[1]))
    for i in range(20):
        proba += estimate_xgb(X_train, X_test, y_train, seed=i)
    return proba / 20

def estimate_mean(X_train, X_test, y_train):
    """
    Just return the mean of this feature in the train set
    """
    ret = []
    for i in [0, 1, 2]:
        ret.append(np.ones(X_test.shape[0]) * y_train[:,i].mean())
    return np.array(ret).T

def estimate_nnet(X_train, X_test, y_train):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.utils import np_utils, generic_utils
    from keras.optimizers import SGD, Adam

    nb_classes = y_train.shape[1]

    dims = X_train.shape[1]

    N = 32

    model = Sequential()
    model.add(Dense(dims, N, init='glorot_uniform'))
    model.add(PReLU((N,)))
    model.add(BatchNormalization((N,)))
    model.add(Dropout(0.5))

    model.add(Dense(N, N/2, init='glorot_uniform'))
    model.add(PReLU((N/2,)))
    model.add(BatchNormalization((N/2,)))
    model.add(Dropout(0.5))

    model.add(Activation('softmax'))
    model.add(Dense(N/2, nb_classes, init='glorot_uniform'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
    print('Compiling...')
    model.compile(loss='mse', optimizer=adam)
    print('Fitting...')
    model.fit(X_train, y_train, verbose=0)
    return model.predict_proba(X_test, verbose=0)

def calc_score(estimator, X, y, train, test):
    proba = estimator(X[train], X[test], y[train])
    each = np.mean((proba - y[test])**2, axis=0) / y[test].mean(axis=0)**2
    return each.mean()

def perform_crossval(estimator, X, y):
    from sklearn.cross_validation import KFold
    skf = KFold(n=y.shape[0], n_folds=10)
    results = Parallel(n_jobs=10)(
            delayed(calc_score)(estimator, X, y, train, test) for train, test in skf
    )
    return results

estimators = [
        ('XGBoost single', estimate_xgb    ),
        ('XGBoost 20x',    estimate_xgb_20 ),
        ('Das Netz',       estimate_nnet   ),
        ('Just the mean',  estimate_mean   ),
]

if __name__ == '__main__':
    print("=== PREPROCESSING")
    X, y = load_data('data/train.txt')

    print('=== RUNNING')
    results = []
    for name, est in estimators:
        est = mem.cache(est)
        results.append((name, perform_crossval(est, X, y)))

    print('=== SCORES')
    for n, s in results:
        print(n, '\t', np.mean(s))


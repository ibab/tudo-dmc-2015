from __future__ import absolute_import
from __future__ import print_function

from functools import partial
import numpy as np
import pandas as pd
from joblib import Memory
from joblib import Parallel, delayed
mem = Memory(cachedir='.joblib', verbose=0)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(1337) # for reproducibility

@mem.cache
def calc_cov(df, k):
    print('Calculate user-user cov {}'.format(k))
    tmp = df.pivot_table(columns=['userID'], index=['couponID{}'.format(k)], values='coupon{}Used'.format(k), aggfunc=np.sum)
    return tmp.cov()

@mem.cache
def calc_collab_score(df, cov, k):
    """
    Calculate mean couponXUsed for all similar users weighted by correlation
    """
    print('Calculate collab score {}'.format(k))
    vals = []
    for i, r in df.iterrows():
        used = df[(df.userID != r.userID) & (df['couponID{}'.format(k)] == r['couponID{}'.format(k)])].groupby('userID')['coupon{}Used'.format(k)].mean()
        cc = cov[r.userID]

        idx = used.index.intersection(cc.index)

        val = np.sum(used.ix[idx] * cc.ix[idx] / cc.ix[idx].sum())
        if np.isfinite(val):
            vals.append(val)
        else:
            vals.append(-999)
    df['collab_score{}'.format(k)] = vals
    return df

def load_data(path):
    df = pd.read_csv(path, delimiter='|')
    data = df.copy()
    data.orderTime = pd.to_datetime(data.orderTime).astype('int64')/1e9/60
    data.couponsReceived = pd.to_datetime(data.couponsReceived).astype('int64')/1e9/60
    data['deltaT'] = (data.orderTime - data.couponsReceived)
    data['priceSum'] = data.price1 + data.price2 + data.price3

    # Create user-coupon matrix
    for i in [1, 2, 3]:
        cov_ = calc_cov(data, i)
        data = calc_collab_score(data, cov_, i)

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
                 'collab_score1',
                 'collab_score2',
                 'collab_score3',
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

    y = data[['coupon1Used', 'coupon2Used', 'coupon3Used', 'basketValue']].values

    return X, y

def estimate_xgb(X_train, y_train, X_test, seed=0):
    scaler = StandardScaler()

    basket_y = y_train[:,3]
    y_train = y_train[:,:3]

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from rep.classifiers import XGBoostClassifier
    clf = XGBoostClassifier(n_estimators=35,
                            max_depth=4,
                            eta=0.32,
                            subsample=0.80,
                            gamma=18,
                            nthreads=2,
                            verbose=1,
                            random_state=seed+50)

    # Classify coupons one by one
    probs = []
    for i in [0, 1, 2]:
        proba = np.zeros(X_test.shape[0])
        clf.fit(X_train, y_train[:,i])
        proba = clf.predict_proba(X_test)[:,1]
        probs.append(proba)

    # TODO provide better estimate
    probs.append(np.ones(X_test.shape[0]) * basket_y.mean())

    return np.array(probs).T

def estimate_xgb_20(X_train, y_train, X_test):
    proba = np.zeros((X_test.shape[0], y_train.shape[1]))
    for i in range(20):
        proba += estimate_xgb(X_train, y_train, X_test, seed=i+40)
    return proba / 20

def estimate_rf(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    basket_y = y_train[:,3]
    y_train = y_train[:,:3]

    probas = []
    for i in [0, 1, 2]:
        clf = RandomForestClassifier(n_estimators=20, max_depth=4)
        clf_ = BaggingClassifier(clf, max_samples=0.7, max_features=0.7)
        clf_.fit(X_train, y_train[:,i])
        probas.append(clf_.predict_proba(X_test)[:,1])
    probas.append(np.ones(X_test.shape[0]) * basket_y.mean())
    return np.array(probas).T

def estimate_mean(X_train, y_train, X_test):
    """
    Just return the mean of this feature in the train set
    """
    ret = []
    for i in [0, 1, 2, 3]:
        ret.append(np.ones(X_test.shape[0]) * y_train[:,i].mean())
    return np.array(ret).T

def estimate_mix(X_train, y_train, X_test):
    p1 = estimate_xgb_20(X_train, y_train, X_test)
    p2 = estimate_nnet(X_train, y_train, X_test)
    return (p1 + p2) / 2

def estimate_nnet(X_train, y_train, X_test):
    import theano.tensor as T
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.utils import np_utils, generic_utils
    from keras.optimizers import SGD, Adam

    basket_y = y_train[:,3]
    y_train = y_train[:,:3]

    nb_classes = y_train.shape[1]
    dims = X_train.shape[1]

    N = 128

    model = Sequential()
    model.add(Dense(dims, N, init='glorot_uniform'))
    model.add(PReLU((N,)))
    model.add(BatchNormalization((N,)))
    model.add(Dropout(0.5))

    model.add(Dense(N, N/2, init='glorot_uniform'))
    model.add(PReLU((N/2,)))
    model.add(BatchNormalization((N/2,)))
    model.add(Dropout(0.3))

    model.add(Activation('softmax'))
    model.add(Dense(N/2, nb_classes, init='glorot_uniform'))

    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
    model.compile(loss='mse', optimizer=adam)
    model.fit(X_train, y_train, verbose=0)

    # TODO provide better estimate
    proba = model.predict_proba(X_test, verbose=0)
    ret = np.hstack([proba, np.ones((X_test.shape[0], 1)) * basket_y.mean()])

    return ret

def calc_score(estimator, X_train, X_test, y_train, y_test):
    proba = estimator(X_train.copy(), y_train.copy(), X_test.copy())
    each = np.mean((proba - y_test)**2, axis=0) / y_test.mean(axis=0)**2
    return each[:3].sum(), each[3]

def perform_crossval(estimator, X, y):
    from sklearn.cross_validation import KFold
    skf = KFold(n=y.shape[0], n_folds=10)
    results = Parallel(n_jobs=10)(
            delayed(calc_score)(estimator, X[train], X[test], y[train], y[test]) for train, test in skf
    )
    return results

estimators = [
        #('Just the mean',  estimate_mean   ),
        #('Das Netz',       estimate_nnet   ),
        ('XGBoost single', estimate_xgb    ),
        #('XGBoost 20x',    estimate_xgb_20 ),
        #('RandomForest',   estimate_rf     ),
        #('Big mix',        estimate_mix    ),
]

if __name__ == '__main__':
    print("=== PREPROC ===")
    X, y = load_data('data/train.txt')

    print('=== RUNNING ===')
    results = []
    for name, est in estimators:
        est = mem.cache(est)
        results.append((name, perform_crossval(est, X, y)))

    print('=== SCORES ===', '\t', 'class', '\t\t', 'regr', '\t\t', 'sum')
    for n, s in results:
        cls, reg = np.mean(s, axis=0)
        print(n, '\t', cls, '\t', reg, '\t', cls + reg)


from __future__ import absolute_import
from __future__ import print_function

from functools import partial, reduce
import numpy as np
import pandas as pd
from joblib import Memory
from joblib import Parallel, delayed
mem = Memory(cachedir='.joblib', verbose=0)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

np.random.seed(1337) # for reproducibility

variables = ['orderID',
             'orderTime',
             'couponsReceived',
             'deltaT',
             'reward1',
             'reward2',
             'reward3',
             'price1',
             'price2',
             'price3',
             'priceSum',
             'basePrice1',
             'basePrice2',
             'basePrice3',
             'premiumProduct1',
             'premiumProduct2',
             'premiumProduct3',
            ]

cat = [('userID',),
       ('brand1',
       'brand2',
       'brand3'),
       ('couponID1',
       'couponID2',
       'couponID3'),
       ('productGroup1',
       'productGroup2',
       'productGroup3'),
       ('categoryIDs1',
       'categoryIDs2',
       'categoryIDs3'),
]

def load_data(path):
    df = pd.read_csv(path, delimiter='|')
    data = df.copy()
    # Remove outliers
    data = data.query('basketValue < 30000').copy()

    data.orderTime = pd.to_datetime(data.orderTime).astype('int64')/1e9/60
    data.couponsReceived = pd.to_datetime(data.couponsReceived).astype('int64')/1e9/60
    data['deltaT'] = (data.orderTime - data.couponsReceived)
    data['priceSum'] = data.price1 + data.price2 + data.price3

    X_var = data[variables].astype(np.float32).values
    X_cat = data[list(reduce(lambda x, y: x + y, cat))]

    X = np.hstack([X_var, X_cat])

    y = data[['coupon1Used', 'coupon2Used', 'coupon3Used', 'basketValue']].values

    return X, y

def default_prep(X_train, X_test):

    ## Scale numeric values
    scaler = StandardScaler()
    scaler.fit(X_train[:,:len(variables)].astype(np.float32))

    X_var_train = scaler.transform(X_train[:,:len(variables)].astype(np.float32))
    X_var_test = scaler.transform(X_test[:,:len(variables)].astype(np.float32))

    #X_var_train = X_train[:,:len(variables)].astype(np.float32)
    #X_var_test = X_test[:,:len(variables)].astype(np.float32)

    X_cat_train = X_train[:,len(variables):].astype('str')
    X_cat_test = X_test[:,len(variables):].astype('str')

    categories_train = dict()
    encoders = dict()
    i = 0
    for group in cat:
        dat = []
        for c in group:
            train = X_cat_train[:,i]
            dat.append(train)
            i += 1
        categories_train[group] = np.concatenate(dat)

    i = 0
    X_cat_train_res = []
    X_cat_test_res = []
    for group in cat:
        encoder = LabelEncoder()
        encoder.fit(categories_train[group])
        encoder.classes_ = np.append(encoder.classes_, ['unknown'])
        for c in group:
            X_cat_train_res.append(encoder.transform(X_cat_train[:,i]))
            contained = np.in1d(X_cat_test[:,i], categories_train[group])
            X_cat_test[:,i][~contained] = 'unknown'
            X_cat_test_res.append(encoder.transform(X_cat_test[:,i]))
            i += 1

    X_train_res = np.vstack(X_cat_train_res).T
    X_test_res = np.vstack(X_cat_test_res).T

    encoder2 = OneHotEncoder(sparse=False)
    encoder2.fit(np.vstack([X_train_res, X_test_res]))
    X_train_res = encoder2.transform(X_train_res)
    X_test_res = encoder2.transform(X_test_res)

    X_train = np.hstack([X_var_train, X_train_res])
    X_test = np.hstack([X_var_test, X_test_res])

    return X_train, X_test

def estimate_ffm(X_train, y_train, X_test, seed=0):
    from tempfile import mkstemp
    from sh import ffm_train, ffm_predict, echo

    X_train, X_test = default_prep(X_train, X_test)

    np.random.seed(seed)
    Q = np.hstack([X_train, y_train])
    np.random.shuffle(Q)
    X_train = Q[:,:-4]
    y_train = Q[:,-4:]

    results = []

    for i in [1,2,3]:
        X = X_train[:,len(variables):]
        y = y_train[:,i-1]
        _, fname = mkstemp(suffix='.txt')
        with open(fname, 'w') as f:
            for xx, yy in zip(X, y):
                f.write(str(yy) + ' ')
                for k, x in enumerate(xx):
                    f.write('{}:{}:1 '.format(k, int(x)))
                f.write('\n')

        mod = fname + '.model'
        log = ffm_train(['-l', '0.008',
                         '-k', '9',
                         '-t', '30',
                         '-r', '0.27',
                         fname,
                         mod,
                        ], _out='/dev/null')

        _, val = mkstemp('.validate.txt')
        with open(val, 'w') as f:
            for xx in X_test[:,len(variables):]:
                f.write('1 ')
                for k, x in enumerate(xx):
                    f.write('{}:{}:1 '.format(k, int(x)))
                f.write('\n')

        _, out = mkstemp('.out.txt')
        ffm_predict([val, mod, out], _out='/dev/null')
        proba = np.loadtxt(out) * np.mean(y_train[:,i-1]) * 3.87
        results.append(proba)

    results.append(np.zeros(X_test.shape[0]))
    return np.array(results).T

def estimate_xgb(X_train, y_train, X_test, seed=0):
    X_train, X_test = default_prep(X_train, X_test)

    basket_y = y_train[:,3]
    y_train = y_train[:,:3]

    #print('Running({})...'.format(X_train.shape))

    #X_train = X_train[:,:len(variables)]
    #X_test = X_test[:,:len(variables)]

    params = [
            {
                'n_estimators': 1300,
                'max_depth': 8,
                'eta': 0.32,
                'subsample': 0.80,
                'gamma': 11,
                'nthreads': 2,
                'verbose': 1,
                'random_state': seed+150
            },
            {
                'n_estimators': 1200,
                'max_depth': 8,
                'eta': 0.30,
                'subsample': 0.80,
                'gamma': 11,
                'nthreads': 2,
                'verbose': 1,
                'random_state': seed+150
            },
            {
                'n_estimators': 1200,
                'max_depth': 8,
                'eta': 0.30,
                'subsample': 0.80,
                'gamma': 11,
                'nthreads': 2,
                'verbose': 1,
                'random_state': seed+150
            },
    ]

    from rep.estimators import XGBoostClassifier

    # Classify coupons one by one
    probs = []
    for i , ps in enumerate(params):
        proba = np.zeros(X_test.shape[0])
        clf = XGBoostClassifier(**ps)
        clf.fit(X_train, y_train[:,i])
        proba = clf.predict_proba(X_test)[:,1]
        probs.append(proba)

    from rep.estimators import XGBoostRegressor
    ps = {
        'n_estimators': 1820,
        'max_depth': 3,
        'eta': 0.305,
        #'subsample': 0.94,
        'colsample': 1.00,
        #'gamma': 0.1,
        'nthreads': 2,
        'verbose': 1,
        'random_state': seed+200
    }
    clf = XGBoostRegressor(**ps)

    #X_train = X_train[:,:len(variables)]
    #X_test = X_test[:,:len(variables)]

    from sklearn.ensemble import GradientBoostingRegressor
    from rep.estimators import XGBoostRegressor
    #from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    #clf = SVR(kernel='rbf', C=50, gamma=0, epsilon=0.1, verbose=False)
    #clf = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0, loss='huber')
    clf.fit(X_train, np.log(basket_y))
    basket_pred = np.exp(clf.predict(X_test))
    #probs.append(np.ones(X_test.shape[0]) * basket_y.mean())
    probs.append(basket_pred)

    return np.array(probs).T

def estimate_xgb_20(X_train, y_train, X_test):
    proba = np.zeros((X_test.shape[0], y_train.shape[1]))
    for i in range(20):
        proba += estimate_xgb(X_train, y_train, X_test, seed=i+40)
    return proba / 20

def estimate_ffm_20(X_train, y_train, X_test):
    proba = np.zeros((X_test.shape[0], y_train.shape[1]))
    for i in range(20):
        proba += estimate_ffm(X_train, y_train, X_test, seed=i)
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

    X_train, X_test = default_prep(X_train, X_test)

    basket_y = y_train[:,3]
    y_train = y_train[:,:3]

    #nb_classes = y_train.shape[1]
    nb_classes = 1
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

    #model.add(Activation('softmax'))
    #model.add(Activation('linear'))
    model.add(Dense(N/2, nb_classes, init='glorot_uniform'))

    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
    model.compile(loss='mse', optimizer=adam)
    model.fit(X_train, basket_y, nb_epoch=300, verbose=0)

    # TODO provide better estimate
    #proba = model.predict_proba(X_test, verbose=0)
    #ret = np.hstack([proba, np.ones((X_test.shape[0], 1)) * basket_y.mean()])
    ret = np.hstack([np.ones((X_test.shape[0], 3)), model.predict(X_test, verbose=0)])

    return ret

def calc_score(estimator, X_train, X_test, y_train, y_test):
    proba = estimator(X_train.copy(), y_train.copy(), X_test.copy())
    each = np.mean((proba - y_test)**2, axis=0) / y_test.mean(axis=0)**2

    return each[0], each[1], each[2], each[3]

def perform_crossval(estimator, X, y):
    from sklearn.cross_validation import KFold
    skf = KFold(n=y.shape[0], n_folds=10)
    results = Parallel(n_jobs=10)(
            delayed(calc_score)(estimator, X[train], X[test], y[train], y[test]) for train, test in skf
    )
    return results

estimators = [
        ('Just the mean',  estimate_mean   ),
        #('Das Netz',       estimate_nnet   ),
        ('XGBoost single', estimate_xgb    ),
        #('XGBoost 20x',    estimate_xgb_20 ),
        #('RandomForest',   estimate_rf     ),
        #('Big mix',        estimate_mix    ),
        #('FFM     ',       estimate_ffm    ),
]

if __name__ == '__main__':
    print("=== PREPROC ===")
    X, y = load_data('data/train.txt')
    #X, y = load_data('tmp.txt')
    X_classify, y_classify = load_data('data/class.txt')

    print('=== RUNNING ===')
    results = []
    for name, est in estimators:
        #est = mem.cache(est)
        results.append((name, perform_crossval(est, X, y)))

    print('=== SCORES ===', '\t', 'c1', '\t\t', 'c2', '\t\t', 'c3', '\t\t', 'csum', '\t\t', 'regr', '\t\t', 'sum')
    for n, s in results:
        c1, c2, c3, reg = np.mean(s, axis=0)
        print(n, '\t', c1, '\t', c2, '\t', c3, '\t', c1+c2+c3, '\t', reg, '\t', c1+c2+c3+reg)

    #for est in estimators:
    #    if est[0] == 'XGBoost 20x':
    #        est = est[1]
    #        break

    #df = pd.read_csv('data/class.txt', delimiter='|')
    #y_pred = est(X, y, X_classify)

    #df['coupon1Used'] = y_pred[:,0]
    #df['coupon2Used'] = y_pred[:,1]
    #df['coupon3Used'] = y_pred[:,2]
    #df['basketValue'] = y_pred[:,3]

    #df = df[['orderID', 'coupon1Used', 'coupon2Used', 'coupon3Used']]

    #df.to_csv('TU_Dortmund_1.txt', sep='|', index=False)



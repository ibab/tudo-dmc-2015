from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.

    Compatible Python 2.7-3.4 

    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.

    Best validation score at epoch 21: 0.4881 

    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)

    Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

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

    for i, r in data.iterrows():
        r['meanCouponResponse1'] = meanResp['coupon1Used'].loc[r.couponID1]

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
                 'meanCouponResponse1',
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

    scaler = StandardScaler()
    scaler.fit(X_var)
    X_var = scaler.transform(X_var)

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

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, y = load_data('data/train.txt')

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

N = 256

model = Sequential()
model.add(Dense(dims, N, init='glorot_uniform'))
model.add(PReLU((N,)))
model.add(BatchNormalization((N,)))
model.add(Dropout(0.5))

model.add(Dense(N, N, init='glorot_uniform'))
model.add(PReLU((N,)))
model.add(BatchNormalization((N,)))
model.add(Dropout(0.5))

model.add(Dense(N, N, init='glorot_uniform'))
model.add(PReLU((N,)))
model.add(BatchNormalization((N,)))
model.add(Dropout(0.5))

model.add(Activation('softmax'))
model.add(Dense(N, nb_classes, init='glorot_uniform'))

def prudsys_score(y_true, y_pred):
        return T.sqr((y_pred - y_true / y_true.mean())).mean()

model.compile(loss='mse', optimizer="adam")

print("Training model...")

model.fit(X, y, nb_epoch=500, batch_size=1024, validation_split=0.10)

print(model.predict(X))

#print("Generating submission...")
#
#proba = model.predict_proba(X_test)
#make_submission(proba, ids, encoder, fname='keras-otto.csv')


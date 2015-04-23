#!/usr/bin/env python2
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.style import use
use('ggplot')

from sklearn.cluster import KMeans, DBSCAN

train = pd.read_csv('./build/train_with_new_features.txt', sep='|')
classify = pd.read_csv('./build/train_with_new_features.txt', sep='|')
df = train.append(classify)

group = df.groupby('userID')

byUser = group.deltaT.agg({
    'meanDeltaT': np.mean,
    'stdDeltaT': np.std,
    'maxDeltaT': np.max,
    'minDeltaT': np.min,
})

byUser = byUser.join(group.priceSum.agg({
    'meanPriceSum': np.mean,
    'stdPriceSum': np.std,
    'maxPriceSum': np.max,
    'minPriceSum': np.min,
}))

byUser = byUser.join(group.basketValue.agg({
    'meanBasketValue': lambda x: np.mean(np.log10(x)),
    'stdBasketValue': lambda x: np.std(np.log10(x)),
    'maxBasketValue': lambda x: np.max(np.log10(x)),
    'minBasketValue': lambda x: np.min(np.log10(x)),
}))

byUser = byUser.join(group.orderTime_minutes.agg({
    'meanOrderTime_minutes': np.mean,
    'stdOrderTime_minutes': np.std,
    'maxOrderTime_minutes': np.max,
    'minOrderTime_minutes': np.min,
}))


byUser.plot(
    kind='scatter',
    x='meanBasketValue',
    y='meanDeltaT',
    s=4,
)
plt.show()

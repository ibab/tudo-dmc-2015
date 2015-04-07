
from pandas import read_csv

df = read_csv('./DMC_2015_orders_train.txt', sep='|')

print(df.describe())


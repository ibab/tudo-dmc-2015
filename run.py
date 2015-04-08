
from pandas import read_csv

df = read_csv('./data/train.txt', sep='|')

print(df.head(5))


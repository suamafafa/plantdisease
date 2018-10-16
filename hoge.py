import pandas as pd
data = pd.read_csv("traindata_int.csv", header=None)
hoge = data.sort_values(0)

import random
def random_no_repeat1(numbers, count):
    number_list = list(numbers)
    random.shuffle(number_list)
    return number_list[:count]

from numpy.random import *
import random                         

df = pd.DataFrame()

ncol=data.shape[0]/30
n = 0
data = data.sort_values(0)
for i in range(int(ncol)):
    start = i*30
    end = (i+1)*30
    hoge = data.iloc[start:end,:]
    #print(hoge.shape)
    ransu = random_no_repeat1(range(30), 3)
    #print(ransu)
    fuga = hoge.iloc[ransu,:]
    if i%10==0:
        print(i, df.shape)
    df = pd.concat([df, fuga], axis=0)
    #print(fuga)

#print("df.shape", df.shape)
df.to_csv("traindata_int_small.csv", header=None, index=None)
print("df.shape", df.shape)

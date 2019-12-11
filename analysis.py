import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw(df, company, model):
    e_min = df['b001100000'].min()
    e_max = df['b001100000'].max()
    p_min = df['b001000000'].min()
    p_max = df['b001000000'].max()
    firm = df.groupby('stkcd')
    f = firm.get_group(company)
    time = f['accper'].values
    f=f.drop(['stkcd','accper'],axis=1)
    f=f.values
    num = f.shape[0] - 5 - 3
    data = f[:, 4]
    testX = []
    if num>=0:
        testX = f[num+3:num+8, :]
    testX = testX.reshape([1, 5, 27])
    pre = model.predict(testX)
    pre_earning = pre[0, :, 0]
    pre_earning *= (e_max - e_min)
    pre_earning += e_min
    data_pre = data[:-3]
    data_pre = np.concatenate((data_pre, pre_earning))
    plt.plot(time, data_pre, "g.-")
    plt.plot(time, data, "r.-")
    plt.show()
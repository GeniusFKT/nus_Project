import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

df = pd.read_csv('D:\\VSCode_Project\\Python\\level13.csv')

def check_na():
    sums = []

    for i in range(31):
        sum = 0
        for j in range(len(df)):
            if pd.isnull(df.iloc[j, i]):
                sum += 1
        sums.append(sum)
        print(df.columns[i], sum)

    x = df.columns
    plt.bar(x, sums)
    plt.show()

def process_date():
    for index, row in df.iterrows():
        if pd.isnull(row['b001200000']):
            print(index)

    # df.to_csv('level13.csv')

process_date()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model

ROOT_DIR = sys.path[0]

class data_analyser():
    def __init__(self, model_path, data_path, input_length, output_length):
        self.raw_df = pd.read_csv(data_path)
        self.input_length = input_length
        self.output_length = output_length
        self.model = load_model(model_path)

        df = self.raw_df.drop(['Unnamed: 0','typrep'],axis=1)
        df = df.fillna(0)
        # df: normalization of raw_df
        cols = df.columns
        normlist=[]
        for col in cols:
            if str(df[col].dtype) == 'float64':
                normlist.append(col)
        for i, name in enumerate(normlist):
            df[name] = (df[name]-df[name].mean())/(df[name].std())

        self.df = df

    def draw(self, company):
        # data used for restoring
        e_mean = self.raw_df['b001100000'].mean()
        e_std = self.raw_df['b001100000'].std()
        p_mean = self.raw_df['b001000000'].mean()
        p_std = self.raw_df['b001000000'].std()

        # data used for drawing
        # data: an ndarray which contains company's profit
        raw_firm = self.raw_df.groupby('stkcd')
        raw_f = raw_firm.get_group(company)
        data = raw_f['b001100000']
        data = data.values

        # time (indicates x axis in the drawing)
        time = raw_f['accper'].values
        func = np.frompyfunc(lambda x: x[:4], 1, 1)
        time = func(time)

        # data used for predicting
        firm = self.df.groupby('stkcd')
        f = firm.get_group(company)
        f = f.drop(['stkcd','accper'],axis=1)
        f = f.values

        num = f.shape[0] - self.input_length - self.output_length
        testX = []
        if num >= 0:
            testX = f[num : num + self.input_length, :]
        else:
            return
        testX = testX.reshape([1, self.input_length, f.shape[1]])

        # predict
        pre = self.model.predict(testX)
        '''
        # restore to unnormalized value
        pre_earning = pre[0, :, 0]
        pre_earning *= e_std
        pre_earning += e_mean
        data_pre = data[:-1]
        data_pre = np.concatenate((data_pre, pre_earning))

        # plot earning
        plt.title("Earning Prediction")
        plt.xlabel("Time")
        plt.ylabel("Earning")
        plt.legend(['predict value', 'actual observed value'], loc=0)
        plt.plot(time, data_pre, "g.-")
        plt.plot(time, data, "r.-")
        plt.savefig('D://VSCode_Project//Python//Project//fig//company%d.png' %(company))
        plt.show()

        '''
        # profit
        data = raw_f['b001000000']
        data = data.values

        # restore
        pre_profit = pre[0, :, 0]
        pre_profit *= p_std
        pre_profit += p_mean
        data_pre = data[:-1]
        data_pre = np.concatenate((data_pre, pre_profit))

        # plot profit
        plt.title("Profit Prediction")
        plt.xlabel("Time")
        plt.ylabel("Profit")
        plt.legend(['predict value', 'actual observed value'], loc=0)
        plt.plot(time, data_pre, "g.-")
        plt.plot(time, data, "r.-")
        plt.savefig('D://VSCode_Project//Python//Project//fig//company%d.png' %(company))
        plt.show()


if __name__ == "__main__":
    analyser = data_analyser("D://VSCode_Project//Python//Project//model//lstm_mae.h5", "D://VSCode_Project//Python//Project//data//level1.csv", 5, 1)
    for i in range(1, 100):
        analyser.draw(i)
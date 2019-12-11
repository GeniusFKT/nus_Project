import numpy as np
import matplotlib.pyplot as plt

class data_analyser():
    def __init__(self, )
    def draw(self, company, model):
        # data used for restoring
        e_min = self.raw_df['b001100000'].min()
        e_max = self.raw_df['b001100000'].max()
        p_min = self.raw_df['b001000000'].min()
        p_max = self.raw_df['b001000000'].max()

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
        testX = testX.reshape([1, self.input_length, f.shape[1]])

        # predict
        pre = model.predict(testX)

        # restore to unnormalized value
        pre_earning = pre[0, :, 0]
        pre_earning *= (e_max - e_min)
        pre_earning += e_min
        data_pre = data[:-3]
        data_pre = np.concatenate((data_pre, pre_earning))

        # plot earning
        plt.title("Earning Prediction")
        plt.xlabel("Time")
        plt.ylabel("Earning")
        plt.legend(['predict value', 'actual observed value'], loc=0)
        plt.plot(time, data_pre, "g.-")
        plt.plot(time, data, "r.-")
        plt.show()

        # profit
        data = raw_f['b001000000']
        data = data.values

        # restore
        pre_profit = pre[0, :, 1]
        pre_profit *= (p_max - p_min)
        pre_profit += p_min
        data_pre = data[:-3]
        data_pre = np.concatenate((data_pre, pre_profit))

        # plot profit
        plt.title("Profit Prediction")
        plt.xlabel("Time")
        plt.ylabel("Profit")
        plt.legend(['predict value', 'actual observed value'], loc=0)
        plt.plot(time, data_pre, "g.-")
        plt.plot(time, data, "r.-")
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class data_loader():
    def __init__(self, data_path, input_length, output_length, ratio):
        self.raw_df = pd.read_csv(data_path)
        self.input_length = input_length
        self.output_length = output_length
        self.ratio = ratio

        df = self.raw_df.drop(['Unnamed: 0','typrep'],axis=1)
        df = df.fillna(0)
        # df: normalization of raw_df
        cols = df.columns
        normlist=[]
        for col in cols:
            if str(df[col].dtype) == 'float64':
                normlist.append(col)
        for i, name in enumerate(normlist):
            df[name] = (df[name]-df[name].min())/(df[name].max()-df[name].min())

        self.df = df

    # fetch training and testing data from data frame
    def get_data(self):
        stkcd=self.df['stkcd']
        stkcd=np.array(stkcd).tolist()
        stkcd=list(set(stkcd))
        firm=self.df.groupby('stkcd')

        dataX,dataY=[],[]
        for i,val in enumerate(stkcd):
            f=firm.get_group(val)
            f=f.drop(['stkcd','accper'],axis=1)
            f=f.values
            num=f.shape[0] - self.input_length - self.output_length
            if num>=0:
                for j in range(num+1):
                    dataX.append(f[j:j+self.input_length,:])
                    dataY.append(np.hstack((f[j+self.input_length:j+self.input_length+self.output_length,4].reshape(3,1), f[j+self.input_length:j+self.input_length+self.output_length,10].reshape(3,1))))
        dataX=np.array(dataX)
        dataY=np.array(dataY)

        train_size = int(len(dataX) * self.ratio)
        trainX = dataX[:train_size]
        trainY = dataY[:train_size]
        testX = dataX[train_size:]
        testY = dataY[train_size:]

        return trainX, trainY, testX, testY

    # Given company's stkcd and training model, 
    # draw a graph of its origin earning and profit
    # together with predicting data of three years
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

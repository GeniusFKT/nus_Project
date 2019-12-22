import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import os

ROOT_DIR = os.path.abspath("..")
FIG_DIR = os.path.join(ROOT_DIR, "fig")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "data")

n_steps_in, n_steps_out = 18, 3


class data_analyser():
    '''
        ## class data_analyser:
        Visualize prediction data by invoking draw function.

        Further functions will be developed in future use
    '''
    def __init__(self, model_path: str, data_path: str, input_length: int, output_length: int):
        '''
            param:
            model_path: model storage path
            data_path: data storage path
            input_length: num of years for training input
            output_length: num of years for training output
        '''
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

    def draw(self, company: int) -> None:
        '''
            draw the graph by given company name
        '''
        # data used for restoring
        e_mean = self.raw_df['b001100000'].mean()
        e_std = self.raw_df['b001100000'].std()
        p_mean = self.raw_df['b001000000'].mean()
        p_std = self.raw_df['b001000000'].std()

        # data used for drawing
        # data: an ndarray which contains company's profit
        raw_firm = self.raw_df.groupby('stkcd')
        try:
            raw_f = raw_firm.get_group(company)
        except KeyError:
            return

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
        data_pre = data[:-self.output_length]
        data_pre = np.concatenate((data_pre, pre_profit))

        # plot profit

        plt.title("Company%d Profit Prediction" %(company))
        plt.xlabel("Time")
        plt.ylabel("Profit")
        plt.plot(time, data_pre, "g.-")
        plt.plot(time, data, "r.-")
        plt.legend(["predict value", "actual observed value"], loc=0)

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 9)
        fig_name = "company%d_mse.png" %(company)
        plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=100)

        plt.show()
        plt.close()


if __name__ == "__main__":
    model_name = os.path.join(MODEL_DIR, "lstm_mae_yc.h5")
    data_name = os.path.join(DATA_DIR, "level1.csv")
    analyser = data_analyser(model_name, data_name, n_steps_in, n_steps_out)
    for i in range(1, 999):
        analyser.draw(i)
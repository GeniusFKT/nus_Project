import numpy as np
import pandas as pd

class data_loader():
    def __init__(self, data_path, input_length, output_length, ratio):
        '''
            PARAM:
            data_path: path to data file
            input_length(int): how many years of data used for prediction
            output_length(int): how many years we need to predict
            ratio: used in get_data v1 version, how many data used for training
        '''

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
            df[name] = (df[name]-df[name].mean())/(df[name].std())

        self.df = df

    def get_data_v1(self):
        '''
            fetch training and testing data from data frame
            Given input_length years data, predict output_length years
            training dataset: former 70% (order by stkcd)
            testing dataset: later 30%
        '''

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

    def get_data(self):
        '''
            Given input_length years data, predict output_length year
            Test dataset is the last input_length + output_length years stat from company (2000+)
        '''

        stkcd=self.df['stkcd']
        stkcd=np.array(stkcd).tolist()
        stkcd=list(set(stkcd))
        firm=self.df.groupby('stkcd')

        trainX,trainY=[],[]
        testX,testY=[],[]

        for i,val in enumerate(stkcd):
            f=firm.get_group(val)
            f=f.drop(['stkcd','accper'],axis=1)
            f=f.values
            num=f.shape[0]-self.input_length-self.output_length
            if (num>=0):
                testX.append(f[-self.input_length-self.output_length:-self.output_length,:])
                testY.append(f[-self.output_length:,10])
                for j in range(num):
                    trainX.append(f[j:j+self.input_length,:])
                    trainY.append(f[j+self.input_length:j+self.input_length+self.output_length,10])
        trainX=np.array(trainX)
        trainY=np.array(trainY)
        testX=np.array(testX)
        testY=np.array(testY)

        return trainX, trainY, testX, testY

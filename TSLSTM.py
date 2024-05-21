from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

class TimeSeriesLSTM:
    def __init__(self, drop_prob, activation):
        super(TimeSeriesLSTM, self).__init__()
        self.model = Sequential()
        self.drop_prob = drop_prob
        self.activation = activation

    def create_model(self):
        # 多层LSTM模型
        self.model.add(LSTM(50, input_dim=1, return_sequences=True))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(200, return_sequences=True))
        self.model.add(LSTM(300, return_sequences=False))
        self.model.add(Dropout(self.drop_prob))
        self.model.add(Dense(100))
        self.model.add(Dense(1))
        # 激活函数选择
        self.model.add(Activation(self.activation))
        return self.model

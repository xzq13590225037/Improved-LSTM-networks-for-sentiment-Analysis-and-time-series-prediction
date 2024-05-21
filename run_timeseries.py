import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from TSLSTM import TimeSeriesLSTM
from utils import create_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--drop_prob', type=str, default='0.2')
    parser.add_argument('--input', type=str, default='StockPrice')
    parser.add_argument('--output', type=str, default='./result/')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()

    # 数据集读取与预处理
    dataframe = pd.read_csv(f'./data/{args.input}.csv', header=0,
                            parse_dates=[0], index_col=0, usecols=[0, 1])
    dataset = dataframe.values

    # 数据可视化
    plt.figure(figsize=(12, 8))
    dataframe.plot()
    plt.ylabel(args.input)
    print(f'Showing {args.input} data')
    plt.show()

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    # 数据集划分
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0: train_size], dataset[train_size: len(dataset)]
    # 生成训练集和测试集
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # 构建 LSTM 模型
    lstm_model = TimeSeriesLSTM(float(args.drop_prob), args.activation)
    model = lstm_model.create_model()
    model.compile(loss='mean_squared_error', optimizer=args.optimizer)

    # 训练模型
    print('Training model...')
    history = model.fit(trainX, trainY, batch_size=args.batch_size, epochs=args.epoch,
                        validation_split=args.validation_split, verbose=args.verbose)

    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 保存模型
    model.save(f'{args.output}{args.input}_model.keras')
    print(f'Model saved to {args.output}{args.input}_model.keras')

    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # 计算评分
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Score %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    print('Test Score %.2f RMSE' % (testScore))

    # 绘图
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot = np.reshape(trainPredictPlot, (dataset.shape[0], 1))
    trainPredictPlot[look_back: len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot = np.reshape(testPredictPlot, (dataset.shape[0], 1))
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1: len(dataset) - 1, :] = testPredict

    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    print(f'Showing loss data')
    plt.savefig(f'{args.output}{args.input}_loss.png')
    plt.show()

    plt.figure(figsize=(20, 15))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.ylabel(f'{args.input}')
    plt.xlabel('date')
    print(f'Showing {args.input} predict data where blue is real data, orange is train data and green is test data')
    plt.savefig(f'{args.output}{args.input}_predict.png')
    plt.show()
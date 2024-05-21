import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from BiLSTM import BiLSTM
from LSTM import LSTM
from utils import tockenize, predict_text, acc

# 数据加载和预处理
def load_and_preprocess_data(df):
    X, y = df['review'].values, df['sentiment'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
    print('train size:{},test size:{}'.format(len(x_train), len(x_test)))
    print(25 * '=', "正在标记数据集", 25 * '=')
    # 确保 review 文本是字符串格式
    x_train = [str(review) for review in x_train]
    x_test = [str(review) for review in x_test]
    x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)

    # 统计每个类别的数量
    dd = pd.Series(y_train).value_counts().sort_index()
    # 获取所有类别的名称
    sentiment_labels = ['negative', 'positive']
    # 创建一个与类别计数相对应的索引数组，确保顺序与 sentiment_labels 匹配
    index = np.arange(len(sentiment_labels))

    # 绘制条形图，x 轴为索引，y 轴为计数
    plt.bar(index, dd.values, tick_label=sentiment_labels)
    plt.ylabel('Count')  # 设置 y 轴标签
    plt.title('Sentiment Distribution')  # 设置图表标题
    plt.show()

    return x_train, x_test, y_train, y_test, vocab

# 结果可视化
def visualize_results(epoch_tr_loss, epoch_vl_loss, epoch_tr_acc, epoch_vl_acc):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train Loss')
    plt.plot(epoch_vl_loss, label='Validation Loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(f'{args.output_path}{args.input_data}_loss_acc.png')
    plt.show()

# 模型训练和评估
def train_and_evaluate(model, train_loader, valid_loader, epochs, clip, batch_size, device, lr):
    # 训练过程
    # loss and optimization functions
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    print(25 * '=', "开始训练", 25 * '=')
    valid_loss_min = np.Inf  # 初始化验证损失最小值为正无穷大

    epoch_tr_loss, epoch_vl_loss = [], []  # 初始化训练损失和验证损失列表
    epoch_tr_acc, epoch_vl_acc = [], []
    # 在训练循环之前初始化上一个epoch的损失
    previous_epoch_loss = np.Inf
    loss_change_threshold = 0.0001  # 设置损失变化的阈值

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()  # 设置模型为训练模式
        h = model.init_hidden(batch_size)  # 初始化隐藏层状态
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签转换为设备上的张量
            h = tuple([each.data for each in h])  # 将隐藏层状态转换为元组,tuple作用是将列表转换为元组
            model.zero_grad()  # 清空梯度缓存
            output, h = model(inputs, h)  # 将输入和隐藏层状态传递给模型,获取输出和隐藏层状态
            loss = criterion(output.squeeze(), labels.float())  # 计算损失
            loss.backward()
            train_losses.append(loss.item())  # 将损失添加到训练损失列表
            accuracy = acc(output, labels)  # 计算准确率
            train_acc += accuracy  # 将准确率添加到训练准确率列表
            # clip_grad_norm：有助于防止 LSTM 中出现梯度爆炸问题。
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()  # 更新模型参数

        # 同理,计算验证损失和准确率
        val_h = model.init_hidden(batch_size)
        val_losses = []
        val_acc = 0.0
        model.eval()  # 设置模型为评估模式
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])
            inputs, labels = inputs.to(device), labels.to(device)
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            accuracy = acc(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)  # 计算平均训练损失
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)  # 计算平均训练准确率
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)  # 将平均训练损失添加到训练损失列表
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)  # 将平均训练准确率添加到训练准确率列表
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch + 1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
        if epoch_val_loss <= valid_loss_min:  # 如果验证损失小于等于最小验证损失,则保存模型
            torch.save(model.state_dict(), f'{args.output_path}{args.input_data}.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            epoch_val_loss))  # 输出验证损失下降的信息
            valid_loss_min = epoch_val_loss  # 更新最小验证损失

        print(25 * '==')  # 输出分隔符
        # 比较当前epoch的损失与上一个epoch的损失
        if epoch > 0 and (abs(previous_epoch_loss - epoch_val_loss) < loss_change_threshold or epoch_val_loss < 0.01):
            print(f'Loss change is below threshold {loss_change_threshold}, stopping training.')
            break  # 中止训练
        # 更新上一个epoch的损失为当前epoch的损失
        previous_epoch_loss = epoch_val_loss

    # 可视化训练过程
    visualize_results(epoch_tr_loss, epoch_vl_loss, epoch_tr_acc, epoch_vl_acc)

if __name__ == '__main__':
    # Set Argument Parser
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--clip', type=int, default=5, help='gradient clipping防止梯度爆炸')
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--embedding_dim', type=str, default='32')
    parser.add_argument('--hidden_dim', type=str, default='128')
    parser.add_argument('--drop_prob', type=str, default='0.5')
    parser.add_argument('--input_data', type=str, default='IMDB')
    parser.add_argument('--output_path', type=str, default='./result/sentiment/LSTM/Adam/')
    args = parser.parse_args()

    ## 数据预处理
    # 导入数据
    base_csv = f'./data/{args.input_data}.csv'
    df = pd.read_csv(base_csv)
    x_train, x_test, y_train, y_test, vocab = load_and_preprocess_data(df)

    # 初始化模型和设备
    model = LSTM(vocab_size=len(vocab) + 1, hidden_dim=int(args.hidden_dim), embedding_dim=int(args.embedding_dim),
                   output_dim=int(args.output_dim), n_layers=int(args.num_layer), drop_prob=float(args.drop_prob))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 以tensor格式载入数据集
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=True)

    # 训练和评估模型
    train_and_evaluate(model, train_loader, valid_loader, args.epoch, args.clip, args.batch_size, device, args.lr)

    # # 实验结果验证
    # print("验证评论情感分析")
    # # 取出两个例子测试预测结果
    # index = 1059  # 设置索引
    # print(df['review'][index])  # 打印第 index 个评论
    # print('=' * 70)
    # print(f'实际情感是: {df["sentiment"][index]}')  # 打印第 index 个评论的实际情感
    # pro = predict_text(df['review'][index], vocab, model, device)  # 预测第 index 个评论的情感
    # status = "positive" if pro > 0.5 else "negative"
    # pro = (1 - pro) if status == "negative" else pro  # 将预测的概率值转换为 0 到 1 之间的数值
    # print(f'预测情感是： {status}，可能性为： {pro}')
    #
    # print('=' * 70)
    # index = 321
    # print(df['review'][index])
    # print('=' * 70)
    # print(f'实际情感是: {df["sentiment"][index]}')
    # print('=' * 70)
    # pro = predict_text(df['review'][index], vocab, model, device)
    # status = "positive" if pro > 0.5 else "negative"
    # pro = (1 - pro) if status == "negative" else pro
    # print(f'预测情感是： {status}，可能性为： {pro}')

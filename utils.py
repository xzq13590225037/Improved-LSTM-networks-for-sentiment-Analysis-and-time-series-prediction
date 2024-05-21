import re
import numpy as np
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter

# 数据清洗
def preprocess_string(s):
    # 删除所有非单词字符（除数字和字母外的所有字符）
    s = re.sub(r"[^\w\s]", '', s)
    # 将所有空格替换为无空格
    s = re.sub(r"\s+", '', s)
    # 用无空格的数字代替
    s = re.sub(r"\d", '', s)
    return s

def tockenize(x_train, y_train, x_val, y_val):
    word_list = []  # 初始化词汇表
    stop_words = set(stopwords.words('english'))  # 获取停用词
    for sent in tqdm(x_train):  # 遍历训练集
        for word in sent.lower().split():  # 遍历单词
            word = preprocess_string(word)  # 预处理
            if word not in stop_words and word != '' and word != ' ':  # 去除停用词
                word_list.append(word)  # 将单词存储在word_list中
    corpus = Counter(word_list)  # 统计词频
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]  # 取前1000个词频最高的词,进行排序

    onehot_dict = {w: i for i, w in enumerate(corpus_)}  # 建立词到数字的映射{词:数字}的字典

    final_list_train, final_list_test = [], []  # 初始化训练集和测试集
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
        '''
        再次遍历训练集的每个句子，使用 onehot_dict 将句子中的每个单词转换为对应的整数索引，
        并创建一个标记化列表,同理遍历测试集
        '''
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])
    encoded_train = [1 if label == 'positive' else 0 for label in y_train]  # 将标签转换为二进制数字01
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]  # 将标签转换为二进制数字01

    #序列长度
    seq_len = 500
    # 对训练集和测试集的序列进行填充或截断到seq_len长度
    x_train_padded = padding(final_list_train, seq_len)
    x_test_padded = padding(final_list_test, seq_len)

    # 将填充后的序列转换为NumPy数组
    x_train_array = np.array(x_train_padded)
    y_train_array = np.array(encoded_train)
    x_test_array = np.array(x_test_padded)
    y_test_array = np.array(encoded_test)

    return x_train_array, y_train_array, x_test_array, y_test_array, onehot_dict
    #np.array(final_list_train)：训练集的标记化结果，转换为NumPy数组。
    # np.array(encoded_train)：训练集的二进制标签，转换为NumFPy数组。
    # np.array(final_list_test)：验证集的标记化结果，转换为NumPy数组。
    # np.array(encoded_test)：验证集的二进制标签，转换为NumPy数组。
    # onehot_dict：单词到整数索引的映射字典

# 将每个序列填充到最大长度
def padding(sentences, seq_len):
    features=np.zeros((len(sentences),seq_len),dtype=int)#初始化一个形状为(len(sentences),seq_len)的全零矩阵
    for ii,review in enumerate(sentences):#遍历句子列表
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]#将句子的前seq_len个词的索引存储在矩阵中,不足的部分用零填充
    return features#返回填充后的矩阵

#定义预测函数
def predict_text(text, vocab, model, device):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in vocab.keys()])#将文本转换为单词序列
    word_seq = np.expand_dims(word_seq,axis=0) #将单词序列扩展为二维数组
    pad =  torch.from_numpy(padding(word_seq,500))#将填充后的单词序列转换为设备张量
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])#tuple()函数用于将列表转换为元组
    output, h = model(inputs, h)
    return(output.item())

# 准确率预测
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

# 创建数据集函数
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


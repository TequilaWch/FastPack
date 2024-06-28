from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import warnings
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None, 'display.max_rows', None, 'display.float_format', lambda x: '%.6f' % x)
warnings.filterwarnings("ignore")

# TIME_RANGE = 64  # 时间步长


# 分组
def process_data(df, group_size):
    # 提取独立标签
    unique_labels = df['ProtocolName'].unique()

    grouped_data = []
    grouped_labels = []

    # 根据组大小划分数据集
    for label in unique_labels:
        label_data = df[df['ProtocolName'] == label].iloc[:, :-1].values.tolist()
        label_size = len(label_data)

        while label_size >= group_size:
            grouped_data.append(label_data[:group_size])
            grouped_labels.append(label)
            label_data = label_data[group_size:]
            label_size -= group_size

    # 3. 返回处理后的数据集和标签集
    return grouped_data, grouped_labels


# 数据处理2
def process_data2(df):
    unique_labels = df['ProtocolName'].unique()
    label_num = len(unique_labels)
    data = df.drop(['ProtocolName'], axis=1).values
    labels = df['ProtocolName'].values
    data_reshaped = data.reshape(-1, 9, 9, 1)
    # labels_encoded = to_categorical(labels, num_classes=label_num)
    return data_reshaped, labels,label_num


# 数据处理3
def process_data3(df, range):
    # 1.将其中具有相同(特征A,标签)组合的样本每六条进行组合成一个(6*82)的矩阵，不足六条的样本直接放弃，最后得到n个(6*82)的矩阵和n个标签(n<=3,000,000/6)
    # 2.移除每个矩阵中的特征A所在列，得到n个(6*81)的矩阵
    # 3.将n个矩阵合并，得到一个(n*6*81)的X和(n,)的Y
    grouped_df = df.groupby(['Flow.ID', 'ProtocolName'])
    X_matrices = []
    Y_labels = []
    for group_name, group_data in grouped_df:
        if len(group_data) >= range:
            X_matrix = group_data.iloc[:range, 1:].values
            # print("1:", X_matrix)
            # X_matrix = np.delete(X_matrix, [0], axis=1)
            # print("2:", X_matrix)
            X_matrix = X_matrix[:, :-1]
            # print("3:", X_matrix)
            X_matrices.append(X_matrix)
            Y_labels.append(group_name[1])
            # break
    X = np.array(X_matrices)
    X = X.reshape(-1, range, 81, 1)
    Y = np.array(Y_labels)
    X = X.astype(float)
    # Y = Y.astype(np.float32)
    label_num = len(np.unique(Y))
    # print("X.shape: ", X.shape)
    # print("X[0]: ", X[0])
    # print("X.dtype: ", X.dtype)
    # print("Y.shape: ", Y.shape)
    # print("Y.dtype: ", Y.dtype)
    # print("label_num: ", label_num)
    return X, Y, label_num


# 读取配置信息
def read_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# 处理掉不需要的行和列
def data_resize(data_path):
    # 读取
    app_df = pd.read_csv(data_path)
    # 去除无意义的label列
    app_df = app_df.drop(['Label'], axis=1)
    # 去除不便于标识的字符类型列, 主要是各个IP
    # 此步存疑
    app_df = app_df.drop(['Flow.ID', 'Source.IP', 'Destination.IP', 'L7Protocol'], axis=1)
    # app_df = app_df.drop(['Source.IP', 'Destination.IP', 'L7Protocol'], axis=1)
    # 去除ProtocolName中的SSL、SSL_No_Cent、Unknown、Flow_Not_Found
    column_to_remove = ['SSL', 'SSL_No_Cent', 'Unknown', 'Flow_Not_Found']
    for value in column_to_remove:
        app_df = app_df[app_df['ProtocolName'] != value]
    # 修改Timestamp的值为int类型
    app_df['Timestamp'] = pd.to_datetime(app_df['Timestamp'], format='%d/%m/%Y%H:%M:%S')
    app_df['Timestamp'] = (app_df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # print("df.shape: ", app_df.shape[1])
    # 返回移除后的dataframe 包含标签ProtocolName
    return app_df


# 根据标签划分为不同的np.arrays，最后得到一个三维的np.array
def df_2_array(df, TIME_RANGE):
    # X, Y = process_data(df, TIME_RANGE)
    range = 3
    X, Y, label_num = process_data2(df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.fit_transform(Y_test)
    # print("Y.dtype: ", Y_train.dtype)
    return X_train, X_test, Y_train, Y_test, label_num, range


# 存储
def array_store(array, csv_path):
    np.savetxt(csv_path, array, delimiter=',')
    return


# 接口
def app_cleaner(TIME_RANGE):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": 开始数据读取和处理阶段")
    start = datetime.now()
    config_path = 'config/config.yaml'
    config = read_config(config_path=config_path)
    app_traffic_path = config['app_traffic_data']  # 应用流量数据集
    app_traffic_df = data_resize(app_traffic_path)  # 处理后的数据 datarame
    # X_train, X_test, Y_train, Y_test, label_encoder = df_2_array(app_traffic_df,TIME_RANGE)  # 分组, 时间步长256
    X_train, X_test, Y_train, Y_test, label_num, range = df_2_array(app_traffic_df, TIME_RANGE)
    # print("训练集大小:", len(Y_train))
    # print("测试集大小:", len(Y_test))
    # print("训练集Y:", Y_train)
    # print("测试集Y:", Y_test)
    # print(set(Y_train))
    # print(X_train.shape)
    # print(np.unique(np.concatenate((X_train, X_test), axis=0)).shape)
    label_num = len(np.unique(np.concatenate((Y_train, Y_test), axis=0)))
    end = datetime.now()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": 数据处理成功! 处理阶段共用时", end - start)
    # return X_train, X_test, Y_train, Y_test, label_encoder, label_num
    return X_train, X_test, Y_train, Y_test, label_num, range


if __name__ == "__main__":
    app_cleaner(64)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocol']
# 数据列
feature_columns = ['frame.len', 'frame.cap_len', 'ip.hdr_len', 'ip.dsfield.ecn', 'ip.len', 'ip.frag_offset', 'ip.ttl',
                   'tcp.hdr_len', 'tcp.len', 'tcp.flags.ns', 'tcp.flags.fin', 'tcp.window_size_value',
                   'tcp.urgent_pointer','udp.length']
timestamp = ['timestamp']
label_column = ['app']

df = pd.read_csv('Processed Data/processedData.csv',low_memory=False).drop('Unnamed: 0',axis=1)
df = df.fillna(0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.dropna()
num_apps = len(df['app'].unique())
#

def preprocess_data_7(df, label):
    label = pd.Series(label, name="app")
    df = pd.concat([df, label], axis=1)  # 14行特征 + 5五元组 + 1时间戳 + 1标签
    df_processed = df.groupby(flow_columns, as_index=False).apply(lambda x: x.sort_values('timestamp')).reset_index(
        drop=True)
    newdf = df_processed[feature_columns + flow_columns + timestamp]
    newlabel = df_processed[label_column]  # 1标签
    newX = np.empty((newdf.shape[0], 7, 14))
    time_interval = pd.to_timedelta('30 ms').total_seconds()  # 上下文时间间隔范围
    for i in range(newdf.shape[0]):
        here = newdf.iloc[i][feature_columns]
        if i == 0:
            before1 = newdf.iloc[i][feature_columns]
            before2 = newdf.iloc[i][feature_columns]
            before3 = newdf.iloc[i][feature_columns]
            # 下一条和本条不一样 或 超过时间范围
            if ((not newdf.iloc[i][flow_columns].equals(newdf.iloc[i + 1][flow_columns])) or
                    ((newdf.iloc[i + 1][timestamp].values[0] - newdf.iloc[i][timestamp].values[
                        0]).total_seconds() > time_interval)):
                next1 = newdf.iloc[i][feature_columns]
                next2 = newdf.iloc[i][feature_columns]
                next3 = newdf.iloc[i][feature_columns]
            else:
                next1 = newdf.iloc[i + 1][feature_columns]
                if ((not newdf.iloc[i + 1][flow_columns].equals(newdf.iloc[i + 2][flow_columns])) or
                        ((newdf.iloc[i + 2][timestamp].values[0] - newdf.iloc[i+1][timestamp].values[
                            0]).total_seconds() > time_interval)):
                    next2 = newdf.iloc[i + 1][feature_columns]
                    next3 = newdf.iloc[i + 1][feature_columns]
                else:
                    next2 = newdf.iloc[i + 2][feature_columns]
                    if ((not newdf.iloc[i + 2][flow_columns].equals(newdf.iloc[i + 3][flow_columns])) or
                            ((newdf.iloc[i + 3][timestamp].values[0] - newdf.iloc[i+2][timestamp].values[
                                0]).total_seconds() > time_interval)):
                        next3 = newdf.iloc[i + 2][feature_columns]
                    else:
                        next3 = newdf.iloc[i + 3][feature_columns]

        # 最后一行没有下一条
        elif i == newdf.shape[0] - 1:
            next1 = newdf.iloc[i][feature_columns]
            next2 = newdf.iloc[i][feature_columns]
            next3 = newdf.iloc[i][feature_columns]
            # 上一条和本条不一样 或 超过时间范围
            if ((not newdf.iloc[i][flow_columns].equals(newdf.iloc[i - 1][flow_columns])) or
                    ((newdf.iloc[i][timestamp].values[0] - newdf.iloc[i - 1][timestamp].values[
                        0]).total_seconds() > time_interval)):
                before1 = newdf.iloc[i][feature_columns]
                before2 = newdf.iloc[i][feature_columns]
                before3 = newdf.iloc[i][feature_columns]
            else:
                before1 = newdf.iloc[i - 1][feature_columns]
                if ((not newdf.iloc[i - 1][flow_columns].equals(newdf.iloc[i - 2][flow_columns])) or
                        ((newdf.iloc[i - 1][timestamp].values[0] - newdf.iloc[i - 2][timestamp].values[
                            0]).total_seconds() > time_interval)):
                    before2 = newdf.iloc[i - 1][feature_columns]
                    before3 = newdf.iloc[i - 1][feature_columns]
                else:
                    before2 = newdf.iloc[i - 2][feature_columns]
                    if ((not newdf.iloc[i - 2][flow_columns].equals(newdf.iloc[i - 3][flow_columns])) or
                            ((newdf.iloc[i - 2][timestamp].values[0] - newdf.iloc[i - 3][timestamp].values[
                                0]).total_seconds() > time_interval)):
                        before3 = newdf.iloc[i - 2][feature_columns]
                    else:
                        before3 = newdf.iloc[i - 3][feature_columns]
        # 在中间
        else:
            # 上一条和本条不一样 或 超过时间范围
            if ((not newdf.iloc[i][flow_columns].equals(newdf.iloc[i - 1][flow_columns])) or
                    ((newdf.iloc[i][timestamp].values[0] - newdf.iloc[i - 1][timestamp].values[
                        0]).total_seconds() > time_interval)):
                before1 = newdf.iloc[i][feature_columns]
                before2 = newdf.iloc[i][feature_columns]
                before3 = newdf.iloc[i][feature_columns]
            else:
                before1 = newdf.iloc[i - 1][feature_columns]
                if i - 1 == 0 or ((not newdf.iloc[i - 1][flow_columns].equals(newdf.iloc[i - 2][flow_columns])) or
                    ((newdf.iloc[i - 1][timestamp].values[0] - newdf.iloc[i - 2][timestamp].values[
                        0]).total_seconds() > time_interval)):
                    before2 = newdf.iloc[i - 1][feature_columns]
                    before3 = newdf.iloc[i - 1][feature_columns]
                else:
                    before2 = newdf.iloc[i - 2][feature_columns]
                    if i - 2 == 0 or ((not newdf.iloc[i - 2][flow_columns].equals(newdf.iloc[i - 3][flow_columns])) or
                        ((newdf.iloc[i - 2][timestamp].values[0] - newdf.iloc[i - 3][timestamp].values[
                            0]).total_seconds() > time_interval)):
                        before3 = newdf.iloc[i - 2][feature_columns]
                    else:
                        before3 = newdf.iloc[i - 3][feature_columns]
            # 下一条和本条不一样 或 超过时间范围
            if ((not newdf.iloc[i][flow_columns].equals(newdf.iloc[i + 1][flow_columns])) or
                    ((newdf.iloc[i + 1][timestamp].values[0] - newdf.iloc[i][timestamp].values[
                        0]).total_seconds() > time_interval)):
                next1 = newdf.iloc[i][feature_columns]
                next2 = newdf.iloc[i][feature_columns]
                next3 = newdf.iloc[i][feature_columns]
            else:
                next1 = newdf.iloc[i + 1][feature_columns]
                if  i + 1 == newdf.shape[0] - 1 or ((not newdf.iloc[i + 1][flow_columns].equals(newdf.iloc[i + 2][flow_columns])) or
                    ((newdf.iloc[i + 2][timestamp].values[0] - newdf.iloc[i+1][timestamp].values[
                        0]).total_seconds() > time_interval)) :
                    next2 = newdf.iloc[i + 1][feature_columns]
                    next3 = newdf.iloc[i + 1][feature_columns]
                else:
                    next2 = newdf.iloc[i + 2][feature_columns]
                    if i + 2 == newdf.shape[0] - 1 or ((not newdf.iloc[i + 2][flow_columns].equals(newdf.iloc[i + 3][flow_columns])) or
                        ((newdf.iloc[i + 3][timestamp].values[0] - newdf.iloc[i+2][timestamp].values[
                            0]).total_seconds() > time_interval)):
                        next3 = newdf.iloc[i + 2][feature_columns]
                    else:
                        next3 = newdf.iloc[i + 3][feature_columns]
        combined = np.stack((before3, before2, before1, here, next1, next2, next3))
        newX[i] = combined
    return newX, newlabel


def feature2pcg():
    flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocol']
    # 数据列
    feature_columns = ['frame.len', 'frame.cap_len', 'ip.hdr_len', 'ip.dsfield.ecn', 'ip.len', 'ip.frag_offset', 'ip.ttl',
                    'tcp.hdr_len', 'tcp.len', 'tcp.flags.ns', 'tcp.flags.fin', 'tcp.window_size_value',
                    'tcp.urgent_pointer','udp.length']
    timestamp = ['timestamp']
    label_column = ['app']

    df = pd.read_csv('Processed Data/processedData.csv',low_memory=False).drop('Unnamed: 0',axis=1)
    df = df.fillna(0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()
    num_apps = len(df['app'].unique())
    X = df[feature_columns]
    X_2 = df[flow_columns].reset_index()
    X_3 = df[timestamp].reset_index()
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_1 = pd.DataFrame(data=X, columns=feature_columns).reset_index()
    le = LabelEncoder()
    Y = le.fit_transform(df['app'])
    X1 = pd.concat([X_1, X_2, X_3], axis=1)
    X1 = X1.drop(columns=["index"])
    X, Y = preprocess_data_7(X1, Y)
    np.save("Trainer_Input/x.npy", X)
    np.save("Trainer_Input/y.npy", Y)

        



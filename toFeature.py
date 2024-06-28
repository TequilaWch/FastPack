import numpy as np
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from datetime import datetime
scenario = 'wild'
mypath = 'Data/Wild Test Data'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
apps = np.unique([f.split('_')[0] for f in onlyfiles])
print(apps, len(apps), len(onlyfiles))

sel_apps = apps
sel_app_files = {i: [] for i in sel_apps}

for fname in onlyfiles:
    app_name = fname.split('_')[0]
    if app_name in sel_apps:
        sel_app_files[app_name].append(fname)


# 标识列
flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocol']
# 数据列
columns = ['frame.number', 'frame.time', 'frame.len', 'frame.cap_len',
           'ip.hdr_len','ip.dsfield.ecn', 'ip.len', 'ip.frag_offset',
           'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst',
           'tcp.hdr_len', 'tcp.len', 'tcp.srcport', 'tcp.dstport', 'tcp.flags.ns','tcp.flags.fin',
           'tcp.window_size_value', 'tcp.urgent_pointer', 'tcp.option_kind', 'tcp.option_len',
           'udp.srcport', 'udp.dstport', 'udp.length']


# 提取协议
def get_protocal(row):
    if not pd.isnull(row['tcp.len']):
        return 'TCP'
    elif not pd.isnull(row['udp.length']):
        return 'UDP'
    else:
        return 'Unknown'


# 提取源端
def get_srt_port(row):
    if not pd.isnull(row['tcp.len']):
        return row['tcp.srcport']
    elif not pd.isnull(row['udp.length']):
        return row['udp.srcport']
    else:
        return 'Unknown'


# 提取目的端
def get_dst_port(row):
    if not pd.isnull(row['tcp.len']):
        return row['tcp.dstport']
    elif not pd.isnull(row['udp.length']):
        return row['udp.dstport']
    else:
        return 'Unknown'


# 提取时间戳
def get_timestamp(df, cloumn_name):
    df[cloumn_name] = pd.to_datetime(df[cloumn_name])
    df["timestamp"] = df[cloumn_name].astype(np.int64)
    return df


def compute_flow_features(df):
    flow_features = {}
    flow_features['total_num_pkts'] = len(df)
    pkt_size = df['ip.len'].astype(float)
    flow_features['total_num_bytes'] = pkt_size.sum()
    flow_features['min_pkt_size'] = pkt_size.min()
    flow_features['max_pkt_size'] = pkt_size.max()
    flow_features['mean_pkt_size'] = pkt_size.mean()
    flow_features['std_pkt_size'] = pkt_size.std()
    iat = pd.to_datetime(df['frame.time']).diff(1).dt.total_seconds().iloc[1:]
    flow_features['min_iat'] = iat.min()
    flow_features['max_iat'] = iat.max()
    flow_features['mean_iat'] = iat.mean()
    flow_features['std_iat'] = iat.std()
    flow_features['dur'] = iat.sum()
    return flow_features


def process_df_by_packet(df):
    df['protocal'] = df.apply(lambda row: get_protocal(row), axis=1)
    df['srcport'] = df.apply(lambda row: get_srt_port(row), axis=1)
    df['dstport'] = df.apply(lambda row: get_dst_port(row), axis=1)
    df = get_timestamp(df, 'frame.time')

    flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocol']

    processed_df = df.drop(columns=['frame.number', 'frame.time','ip.proto', 'ip.src', 'ip.dst',
                                    'tcp.srcport', 'tcp.dstport','tcp.option_kind', 'tcp.option_len',
                                    'udp.srcport', 'udp.dstport'
                                    ], axis=1)

    processed_df[flow_columns] = df[['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocal']]

    return processed_df


def clean_up_duplicate(row):
    if len(str(row['ip.hdr_len']).split(',')) > 1:
        row['ip.hdr_len'] = str(row['ip.hdr_len']).split(',')[1]
    if len(str(row['ip.len']).split(',')) > 1:
        row['ip.len'] = str(row['ip.len']).split(',')[1]
    else:
        row['ip.len'] = str(row['ip.len']).split(',')[0]
    if len(row['ip.src'].split(',')) > 1:
        row['ip.src'] = row['ip.src'].split(',')[1]
    if len(row['ip.dst'].split(',')) > 1:
        row['ip.dst'] = row['ip.dst'].split(',')[1]
    try:
        if len(row['ip.ttl'].split(',')) > 1:
             row['ip.ttl'] = row['ip.ttl'].split(',')[1]
    except:
        row['ip.ttl'] = row['ip.ttl']

    try:
        if len(row['ip.dsfield.ecn'].split(',')) > 1:
            row['ip.dsfield.ecn'] = row['ip.dsfield.ecn'].split(',')[1]
    except:
        row['ip.dsfield.ecn'] = row['ip.dsfield.ecn']
    try:
        if len(row['ip.frag_offset'].split(',')) > 1:
            row['ip.frag_offset'] = row['ip.frag_offset'].split(',')[1]
    except:
        row['ip.frag_offset'] = row['ip.frag_offset']

    return row



df_all = pd.DataFrame()
for app in sel_apps:
    integrity = True
    df_app = pd.DataFrame()
    for fname in sel_app_files[app]:
        action = fname.split('_')[1]

        df = pd.read_csv(join(mypath, fname), usecols=columns, low_memory=False)
        df = df[df['ip.src'].notna()]

        df = df.apply(lambda row: clean_up_duplicate(row), axis=1)

        # Remove self loop pkts
        df = df[(df['ip.src'] != '127.0.0.1') & (df['ip.dst'] != '127.0.0.1')]
        try:
            df_packet = process_df_by_packet(df)
            df_packet['action'] = action
            df_app = df_app.append(df_packet)
        except:
            integrity = False
            print('\n Error while processing {}. \n'.format(fname))

    df_app['app'] = app

    if integrity:
        df_all = df_all.append(df_app)

df_shuffled = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_csv('Processed Data/wild_lstm_lstm.csv')  # 打乱了排序
print('Finished processing {} scenario data.'.format(scenario))






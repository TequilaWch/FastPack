#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
@author: Yuqiang (Ethan) Heng
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

scenario = 'deterministic' #deterministic, random or wild
if scenario == 'random':
    mypath = 'Data/Randomized Automated Data'
elif scenario == 'deterministic':
    mypath = 'Data/Deterministic Automated Data'
elif scenario == 'wild':
    mypath = 'Data/Wild Test Data'
else:
    raise NameError('Dataset Not Supported')

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
apps = np.unique([f.split('_')[0] for f in onlyfiles])
# print(apps, len(apps), len(onlyfiles))
app_actions = np.unique(['_'.join(f.split('_')[:2]) for f in onlyfiles])
# print(app_actions, len(app_actions))

sel_apps = apps
sel_app_files = {i:[] for i in sel_apps}

for fname in onlyfiles:
    app_name = fname.split('_')[0]
    if app_name in sel_apps:
        sel_app_files[app_name].append(fname)


flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocal']

def get_protocal(row):
    if not pd.isnull(row['tcp.len']):
        return 'TCP'
    elif not pd.isnull(row['udp.length']):
        return 'UDP'
    else:
        return 'Unknown'
    
def get_srt_port(row):
    if not pd.isnull(row['tcp.len']):
        return row['tcp.srcport']
    elif not pd.isnull(row['udp.length']):
        return row['udp.srcport']
    else:
        return 'Unknown'
    
def get_dst_port(row):
    if not pd.isnull(row['tcp.len']):
        return row['tcp.dstport']
    elif not pd.isnull(row['udp.length']):
        return row['udp.dstport']
    else:
        return 'Unknown'
    
columns = ['frame.number','frame.time','frame.len','frame.cap_len','ip.hdr_len',
           'ip.dsfield.ecn','ip.len','ip.frag_offset','ip.ttl','ip.proto','ip.src',
           'ip.dst','tcp.hdr_len','tcp.len','tcp.srcport','tcp.dstport','tcp.flags.ns',
           'tcp.flags.fin','tcp.window_size_value','tcp.urgent_pointer','tcp.option_kind',
           'tcp.option_len','udp.srcport','udp.dstport','udp.length']

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

def process_df_by_flow(df):
    df['protocal'] = df.apply(lambda row: get_protocal(row), axis=1)
    df['srcport'] = df.apply(lambda row: get_srt_port(row), axis=1)
    df['dstport'] = df.apply(lambda row: get_dst_port(row), axis=1)  
    df_flow = pd.DataFrame()
    flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocal']
    ul_flows = {}
    dl_flows = {}
    for flow, flow_df in df.groupby(by=flow_columns):
        if flow[0].split('.')[0] == '10':
            ul_flows[flow] = compute_flow_features(flow_df)
        else:
            dl_flows[flow] = compute_flow_features(flow_df)
    for ul_flow, ul_flow_features in ul_flows.items():
        for dl_flow, dl_flow_features in dl_flows.items():
            if (ul_flow[0] == dl_flow[2]) & (ul_flow[2] == dl_flow[0]) & (ul_flow[1] == dl_flow[3]) & (ul_flow[3] == dl_flow[1]) & (ul_flow[4] == dl_flow[4]):
                ul_flow_features = {'ul_'+feature_name:feature for feature_name,feature in ul_flow_features.items()}
                dl_flow_features = {'dl_'+feature_name:feature for feature_name,feature in dl_flow_features.items()}
                bi_flow_features = {**ul_flow_features,**dl_flow_features}
                bi_flow_features['ip_A'] = ul_flow[0]
                bi_flow_features['port_A'] = ul_flow[1]
                bi_flow_features['ip_B'] = ul_flow[2]
                bi_flow_features['port_B'] = ul_flow[3]
                bi_flow_features['protocal'] = ul_flow[4]
                df_flow = df_flow.append(bi_flow_features, ignore_index=True)
    return df_flow

def clean_up_duplicate(row):
    if len(str(row['ip.hdr_len']).split(','))>1:
        row['ip.hdr_len'] = str(row['ip.hdr_len']).split(',')[1]
    if len(str(row['ip.len']).split(','))>1:
        row['ip.len'] = str(row['ip.len']).split(',')[1]
    else:
        row['ip.len'] = str(row['ip.len']).split(',')[0]
    if len(row['ip.src'].split(','))>1:
        row['ip.src'] = row['ip.src'].split(',')[1]
    if len(row['ip.dst'].split(','))>1:
        row['ip.dst'] = row['ip.dst'].split(',')[1]
    return row


df_all = pd.DataFrame()
for app in sel_apps:
    integrity = True
    df_app = pd.DataFrame()
    for fname in sel_app_files[app]:
        action = fname.split('_')[1]
        df = pd.read_csv(join(mypath,fname),usecols = columns,low_memory=False)
        df = df[df['ip.src'].notna()]
        
        df = df.apply(lambda row:clean_up_duplicate(row),axis=1)
        
        # Remove self loop pkts
        df = df[(df['ip.src']!='127.0.0.1') & (df['ip.dst']!='127.0.0.1')]
        try:
            df_flow = process_df_by_flow(df)
            df_flow['action'] = action
            df_app = df_app.append(df_flow)
        except:
            integrity = False
            print('\n Error while processing {}. \n'.format(fname))

    df_app['app'] = app
    
    if integrity:
        df_all = df_all.append(df_app)
        
df_all.to_csv('Processed Data/all.csv')
print('Finished processing {} scenario data.'.format(scenario))





from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Flatten, Input, Reshape, TimeDistributed
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocol']
# 数据列
feature_columns = ['frame.len', 'frame.cap_len', 'ip.hdr_len', 'ip.dsfield.ecn', 'ip.len', 'ip.frag_offset', 'ip.ttl',
                   'tcp.hdr_len', 'tcp.len', 'tcp.flags.ns', 'tcp.flags.fin', 'tcp.window_size_value',
                   'tcp.urgent_pointer','udp.length']
timestamp = ['timestamp']
label_column = ['app']
df = pd.read_csv('Processed Data/wild_lstm_lstm.csv',low_memory=False).drop('Unnamed: 0',axis=1)
df = df.fillna(0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.dropna()
num_apps = len(df['app'].unique())


X = df[feature_columns]
X_2 = df[flow_columns].reset_index()
X_3 = df[timestamp].reset_index()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_1 = pd.DataFrame(data=X, columns=feature_columns).reset_index()
le = LabelEncoder()
Y = le.fit_transform(df['app'])
X1 = pd.concat([X_1, X_2, X_3], axis=1)
X1 = X1.drop(columns=["index"])

unit1 = 256
unit2 = 256
PCG = 7
time = "30ms"
Y = np.load(f"y_{time}_{PCG}PCG.npy")
X = np.load(f"x_{time}_{PCG}PCG.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(PCG, 14)),
    Reshape((1, PCG, 14)),
    TimeDistributed(LSTM(units=unit1, return_sequences=True)),
    Dropout(0.1),
    Flatten(),
    Reshape((PCG, unit1)),
    LSTM(units=unit2),
    Dropout(0.1),
    Dense(num_apps, activation="softmax")
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min',
                               restore_best_weights=True)
epochs = 500
model.summary()
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=128, callbacks=[early_stopping])
model.save(f'lstm{unit1}_lstm{unit2}_{time}_{PCG}pcg')

model = load_model(f'lstm{unit1}_lstm{unit2}_{time}_{PCG}pcg')

a, X_train_val, b, Y_train_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

loss1, accuracy1 = model.evaluate(X_train_val, Y_train_val)
print("验证loss: ", loss1)
print("验证accuracy: ", accuracy1)
loss2, accuracy2 = model.evaluate(X_test, Y_test)
print("测试loss: ", loss2)
print("测试accuracy: ", accuracy2)

model.summary()

# # 计算测试集中每个样本的真阳性假阳性率
Y_pred = model.predict(X_train_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_labels = le.inverse_transform(Y_train_val)
Y_pred_labels = le.inverse_transform(Y_pred_classes)
#
# 计算混淆矩阵
conf_matrix = confusion_matrix(Y_test_labels, Y_pred_labels)
num_classes = conf_matrix.shape[0]
precision = precision_score(Y_test_labels, Y_pred_labels, average=None)
recall = recall_score(Y_test_labels, Y_pred_labels, average=None)
accuracy = accuracy_score(Y_test_labels, Y_pred_labels)
print(precision)
print(recall)
print(accuracy)

#
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix of LSTM{unit1}-LSTM{unit2}-{time}-{PCG}PCG')
plt.show()
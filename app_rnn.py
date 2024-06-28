from Data_Process import data_process
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Flatten, Reshape, SimpleRNN,Conv2D,BatchNormalization,Dropout
import yaml
from datetime import datetime, timezone
import csv
import os
import pandas as pd
import numpy as np
from app_process import app_cleaner


# 尝试新模型
def create_model(input_shape,label_num):
    model = Sequential([
        BatchNormalization(),
        Conv2D(32, (3, 3), (1, 1), padding='VALID', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), (1, 1), padding='VALID'),
        BatchNormalization(),
        Flatten(),
        # Reshape((1,-1)),
        # LSTM(100),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dropout(0.4),
        Dense(label_num, activation='softmax')
    ])
    return model


# 训练
def trainer(model, X_train, Y_train):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min',
                                                      restore_best_weights=True)
    epochs = 60
    model.fit(X_train, Y_train, epochs=epochs, callbacks=[early_stopping])
    return model

# rnn 训练
def rnn_trainer(X_train, Y_train, label_num, TIME_RANGE):
    X_train = X_train.astype('float64')
    model = Sequential([
        LSTM(units=32,input_shape=(TIME_RANGE, 81)),
        # Reshape((1, 32)),
        # SimpleRNN(units=32),
        Dense(label_num,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, np.array(Y_train), epochs=10, batch_size=32, validation_split=0.2)
    return model

# 验证
def model_val(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, np.array(Y_test))
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": 模型评估完成，已返回")
    return loss, accuracy


# rnn接口
def app_rnn(TIME_RANGE):
    # X_train, X_test, Y_train, Y_test, label_encoder, label_num = app_cleaner(TIME_RANGE)
    X_train, X_test, Y_train, Y_test, label_num, range = app_cleaner(TIME_RANGE)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": 开始模型训练阶段")
    start = datetime.now()
    input_shape = (9, 9, 1)
    model = create_model(input_shape, label_num)
    model = trainer(model, X_train, Y_train)
    # model = rnn_trainer(X_train, Y_train, label_num, TIME_RANGE)
    model.save("app_traffic_model/app_model.h5")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),": 模型训练完毕,模型存储至%s" % "app_traffic_model/app_model.h5")
    loss, accuracy = model_val(model, X_test, Y_test)
    with open('result.txt','a') as file:
        if file.tell() > 0:
            file.write('\n')
        file.write(str(TIME_RANGE) + '\t' + str(loss) + '\t' + str(accuracy) )

    time = datetime.now() - start
    end = datetime.now()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": 模型训练成功! 训练阶段共用时", end - start)

    return


if __name__ == "__main__":
    # for i in range(8):
    #     TIME_RANGE = i + 1
    #     app_rnn(TIME_RANGE)
    app_rnn(1)
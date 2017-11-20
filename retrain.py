# -*- coding: utf-8 -*-
import keras
import pandas as pd
import numpy as np
import keras.backend as K

from datetime import datetime
from common import precision, recall
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from keras.utils import np_utils


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

COLUMNS = ['WABPm', 'WICPm', 'WHR']  # 使用的欄位名稱
FILE_NAME = 'test.csv'  # 要餵進去的資料
MODEL_PATH = 'models/binary-WICPm-WABPm-WHR.h5'  # 需要加強訓練的model路徑

# Load testing data
train_df = pd.read_csv(FILE_NAME)
train_X = train_df.loc[:, COLUMNS].as_matrix()
train_Y = train_df.loc[:, 'ICP_condition'].as_matrix()
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_Y)
encoded_Y = encoder.transform(train_Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# Load trained model
model = keras.models.load_model(MODEL_PATH, custom_objects={
                                'recall': recall,
                                'precision': precision,
                                'mean_pred': mean_pred,
                                })
# Retrain this model
model.fit(train_X, dummy_y,
          epochs=20,
          batch_size=128)
# Save trained model
model.save('models/model-%s.h5' % datetime.isoformat(datetime.now()))
score2 = precision_recall_fscore_support(np.argmax(model.predict(train_X), axis=1), np.argmax(dummy_y, axis=1), average='macro')
print
print 'Precision: %s, Recall: %s, Fscore: %s' % score2[:3]
print 'DONE'

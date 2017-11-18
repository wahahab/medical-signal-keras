import keras
import pandas as pd

from datetime import datetime

COLUMNS = ['WABPm', 'WICPm']  # 使用的欄位名稱
FILE_NAME = 'test.csv'  # 要餵進去的資料
MODEL_PATH = 'models/binary-WICPm-WABPm-WHR.h5'  # 需要加強訓練的model路徑

# Load testing data
train_df = pd.read_csv(FILE_NAME)
train_X = train_df.loc[:, COLUMNS].as_matrix()
train_Y = train_df.loc[:, 'ICP_condition'].as_matrix()
# Load trained model
model = keras.models.load_model(MODEL_PATH)
# Retrain this model
model.fit(train_X, train_Y,
          epochs=20,
          batch_size=128)
# Save trained model
model.save('models/model-%s.h5' % datetime.isoformat(datetime.now()))
print 'DONE'

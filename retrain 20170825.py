import keras
import sys
import pandas as pd
import time
# from datetime import datetime, time

# Load testing data
new = 'Y04407_WI'        # ---------------- enter the filename of test model 
#train_df = pd.read_csv('test.csv' if sys.argv < 2 else sys.argv[1])
train_df = pd.read_csv(new + '.csv')
train_X = train_df.loc[:, ['WABPm', 'WICPm']].as_matrix()
train_Y = train_df.loc[:, 'ICP_condition'].as_matrix()

# Load trained model
trained = 'Retrain model adding W91419_WI-201708251610.h5' # ---------------- enter the filename of trained model 
model = keras.models.load_model(trained)

# Retrain this model
model.fit(train_X, train_Y,
          epochs=20,
          batch_size=128)
# Save trained model
# model.save('model-%s.h5' % datetime.isoformat(datetime.now()))
fmt = '%Y%m%d%H%M'
now = time.localtime()
model.save('Retrain model adding ' + new + '-%s.h5' % time.strftime(fmt, now))
print('DONE')

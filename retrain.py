import keras
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load testing data
train_df = pd.read_csv('test.csv' if sys.argv < 2 else sys.argv[1])
train_X = train_df.loc[:, ['WABPm', 'WICPm']].as_matrix()
train_Y = train_df.loc[:, 'ICP_condition'].as_matrix()
# Load trained model
model = keras.models.load_model('model.h5')
# Retrain this model
model.fit(train_X, train_Y,
          epochs=20,
          batch_size=128)
# Save trained model
model.save('model.h5')
print 'DONE'

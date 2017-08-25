import keras
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

# Load testing data
# test_df = pd.read_csv('test.csv' if sys.argv < 2 else sys.argv[1])
test = 'W91419_WI'
test_df = pd.read_csv(test + '.csv')

test_X = test_df.loc[:, ['WABPm', 'WICPm']].as_matrix()
test_Y = test_df.loc[:, 'ICP_condition'].as_matrix()
print(test_X)
print(test_Y)

# Load trained model
trained = 'model-201708251611.h5'
model = keras.models.load_model(trained)
# score = model.evaluate(test_X, test_Y, batch_size=128)
# print 'Test scores: loss => %s, acc =>%s.' % (score[0], score[1])
# Save test data predict result
predict_Y = model.predict(test_X, batch_size=128)[:, 0]
np.savetxt('test_predict.txt', predict_Y)
# Visualize predict result
plt.plot(range(len(predict_Y)), predict_Y, 'b-',
         range(len(test_Y)), test_Y, 'r,')
plt.ylim(-1, 2)
# plt.show()
savefig('Testing of ' + test + '.png')
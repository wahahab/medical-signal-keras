import sys
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from common import precision, recall
from sklearn.preprocessing import LabelEncoder
# from datetime import datetime
# from keras.utils.np_utils import to_categorical


def train_and_save(X, Y, model_name):
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy', precision, recall])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', precision, recall])

    model.fit(X, Y,
              epochs=5,
              batch_size=128)
    # Save trained model
    # model.save('models/model-%s.h5' % datetime.isoformat(datetime.now()))
    model.save('models/%s.h5' % (model_name))
    score = model.evaluate(X, Y, batch_size=128)
    print
    print 'Good-Bad model training scores: %s' % (score)
    return model

# Load training data
train_df = pd.read_csv('train.csv')
train_df = train_df.loc[train_df.Condition < 7, :]
train_X = train_df.loc[:, ['WICPm']].as_matrix()
train_Y = train_df.loc[:, 'Condition'].as_matrix()
train_Y = train_Y <= 3
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_Y)
encoded_Y = encoder.transform(train_Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

model = train_and_save(train_X, dummy_y, 'WICPm alone')
# perform testing
test_df = pd.read_csv('test.csv' if len(sys.argv) < 2 else sys.argv[1])
test_df = test_df.loc[test_df.ICP_condition < 7, :]
test_X = test_df.loc[:, ['WICPm']].as_matrix()
test_Y = test_df.loc[:, 'ICP_condition'].as_matrix()
predict_Y = model.predict(test_X, batch_size=128)
np.savetxt('binary-all-predict.txt', predict_Y)
# # train impovement prediction model
# good_mask = (train_Y <= 3)
# train2_X = train_X[good_mask, :]
# train2_Y = train_Y[good_mask]
# Y = ((train2_Y == 1) | (train2_Y == 3))
# train_and_save(train2_X, Y, 'improving-model')
# # train deterioration prediction model
# bad_mask = (train_Y > 3)
# train3_X = train_X[bad_mask, :]
# train3_Y = train_Y[bad_mask]
# Y = ((train3_Y == 4) | (train3_Y == 6))
# train_and_save(train3_X, Y, 'deteriorating-model')
#

N = 250

ax = plt.axes()
fig = plt.gcf()
xx, yy = np.meshgrid(np.arange(-50, N, 1),
                     np.arange(-50, N, 1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
# visualize improvement predict result
# good_mask = Z < 0.5
# improving_Z = improve_model.predict(np.c_[xx.ravel(), yy.ravel()])
# improving_Z = improving_Z.reshape(xx.shape)
# deteriorating_Z = deteriorate_model.predict(np.c_[xx.ravel(), yy.ravel()])
# deteriorating_Z = deteriorating_Z.reshape(xx.shape)
# deteriorating_Z[good_mask] = improving_Z[good_mask]
out = ax.contourf(xx, yy, Z, cmap='tab20')
# add color bar to illustrate meaning of contours
# cbar = fig.colorbar(out, ticks=[0, .5, 1])
# cbar.ax.set_title('Probability of bad condition (predict result)', rotation='-90', loc='right', x=3.5)
# print train['Condition'].as_matrix()
# train.sample(300).plot('WABPm', 'WICPm', ax=ax, kind='scatter', c=list(train['Condition']), cmap=plt.get_cmap('ocean'))
# ture_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='w', s=30, marker='.', edgecolors='k', label='BAD')
# false_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='b', s=30, marker='.', edgecolors='k', label='GOOD')
# ax.legend(['BAD', 'GOOD'])
ax.set_title('Model visualization')
plt.show()









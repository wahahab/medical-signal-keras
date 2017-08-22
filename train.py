import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from datetime import datetime


# Load training data
train_df = pd.read_csv('train.csv')
train_X = train_df.loc[:, ['WABPm', 'WICPm']].as_matrix()
train_Y = train_df.loc[:, 'ICP_condition'].as_matrix()

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_X, train_Y,
          epochs=20,
          batch_size=128)
# Save trained model
model.save('models/model-%s.h5' % datetime.isoformat(datetime.now()))
score = model.evaluate(train_X, train_Y, batch_size=128)
print 'Train scores: loss => %s, acc =>%s.' % (score[0], score[1])

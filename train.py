import sys

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
# from common import precision, recall
# from common import mcor, recall, f1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


def cal_metrics(true_val, predict_val):
    t = true_val.sum()
    f = len(true_val) - t
    tp = ((true_val == 1) & (predict_val == 1)).sum() / float(t)
    fn = ((true_val == 1) & (predict_val != 1)).sum() / float(t)
    fp = ((true_val != 1) & (predict_val == 1)).sum() / float(f)
    tn = ((true_val != 1) & (predict_val != 1)).sum() / float(f)
    return tp, fn, fp, tn


def train_and_save(X, Y, model_name):
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='softmax'))
    loss = 'binary_crossentropy' if Y.shape[1] == 2 else 'categorical_crossentropy'
    optimizer = 'rmsprop' if Y.shape[1] == 2 else 'adam'
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, Y,
              epochs=5,
              batch_size=128)
    # Save trained model
    # model.save('models/model-%s.h5' % datetime.isoformat(datetime.now()))
    model.save('models/%s.h5' % (model_name))
    score = model.evaluate(X, Y, batch_size=128)
    predict_val = np.argmax(model.predict(X), axis=1)
    true_val = np.argmax(Y, axis=1)
    score2 = precision_recall_fscore_support(true_val, predict_val, average='macro')
    score3 = cal_metrics(true_val, predict_val)
    print
    print 'Accuracy: %s' % (score[1])
    print 'Precision: %s, Recall: %s, Fscore: %s' % score2[:3]
    print 'tp: %s, fn: %s, fp: %s, tn: %s' % score3
    return model

# # perform testing
# test_df = pd.read_csv('test.csv' if len(sys.argv) < 2 else sys.argv[1])
# test_df = test_df.loc[test_df.ICP_condition < 7, :]
# test_X = test_df.loc[:, ['WICPm']].as_matrix()
# test_Y = test_df.loc[:, 'ICP_condition'].as_matrix()
# predict_Y = model.predict(test_X, batch_size=128)
# np.savetxt('binary-all-predict.txt', predict_Y)

# N = 250

# ax = plt.axes()
# fig = plt.gcf()
# xx, yy = np.meshgrid(np.arange(-50, N, 1),
#                      np.arange(-50, N, 1))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = np.argmax(Z, axis=1)
# Z = Z.reshape(xx.shape)
# # visualize improvement predict result
# # good_mask = Z < 0.5
# # improving_Z = improve_model.predict(np.c_[xx.ravel(), yy.ravel()])
# # improving_Z = improving_Z.reshape(xx.shape)
# # deteriorating_Z = deteriorate_model.predict(np.c_[xx.ravel(), yy.ravel()])
# # deteriorating_Z = deteriorating_Z.reshape(xx.shape)
# # deteriorating_Z[good_mask] = improving_Z[good_mask]
# out = ax.contourf(xx, yy, Z, cmap='tab20')
# # add color bar to illustrate meaning of contours
# # cbar = fig.colorbar(out, ticks=[0, .5, 1])
# # cbar.ax.set_title('Probability of bad condition (predict result)', rotation='-90', loc='right', x=3.5)
# # print train['Condition'].as_matrix()
# # train.sample(300).plot('WABPm', 'WICPm', ax=ax, kind='scatter', c=list(train['Condition']), cmap=plt.get_cmap('ocean'))
# # ture_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='w', s=30, marker='.', edgecolors='k', label='BAD')
# # false_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='b', s=30, marker='.', edgecolors='k', label='GOOD')
# # ax.legend(['BAD', 'GOOD'])
# ax.set_title('Model visualization')
# plt.show()


def start_train(features, isBinary):
    # Load training data
    train_df = pd.read_csv('train.csv')
    train_df = train_df.loc[train_df.Condition < 7, :]
    train_X = train_df.loc[:, features].as_matrix()
    train_Y = train_df.loc[:, 'Condition'].as_matrix()
    train_Y = (train_Y <= 3).astype(int) if isBinary else train_Y
    labels = [1, 0] if isBinary else range(1, 8)
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(train_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y, len(labels))
    model_name = ('binary-' if isBinary else 'multi-') + '-'.join(features)
    model = train_and_save(train_X, dummy_y, model_name)
    return model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('usage: python train.py (--multi) \'WICPm,WABPm,WHR\'')
    isBinary = False if sys.argv[1] == '--multi' else True
    features = sys.argv[1] if not sys.argv[1].startswith('--') else sys.argv[2]
    features = features.split(',')
    model = start_train(features, isBinary)
    sys.exit(0)









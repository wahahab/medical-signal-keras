import numpy
import pandas
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

IS_DEV = True
NB_EPOCH = 1 if IS_DEV else 100
BATCH_SIZE = 1024 if IS_DEV else 5

# load dataset
dataframe = pandas.read_csv("train.csv")
# split into input (X) and output (Y) variables
X = dataframe.loc[:, ['WABPm', 'HR']].values
Y = dataframe.loc[:, 'ICPm'].values


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=NB_EPOCH,
                           batch_size=BATCH_SIZE, verbose=1)

# start training
print 'start training...'
estimator.fit(X, Y)

print 'start predicting...'
# load test dataset
dataframe = pandas.read_csv('test.csv')
X = dataframe.loc[:, ['WABPm', 'HR']].values
# predict ICPm
Y = estimator.predict(X)

# write predict result
print 'writing predict result...'
numpy.savetxt('predict_icp_result.txt', Y)

# save model
# estimator.model.save('icp_estimator.h5')

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

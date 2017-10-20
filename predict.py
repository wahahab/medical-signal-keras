import sys
import pandas as pd
import numpy as np

from os.path import splitext, basename
from keras import models
from common import precision, recall


def perform_predict(model_name, test_file, features):
    model = models.load_model(model_name, {
                                'precision': precision,
                                'recall': recall,
                            })
    test_df = pd.read_csv(test_file)
    X = test_df.ix[:, features].as_matrix()
    predict_result = model.predict(X)
    base_model_name = splitext(basename(model_name))[0]
    base_test_file = splitext(basename(test_file))[0]
    np.savetxt(base_model_name + '-' + base_test_file + '.out', predict_result)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception('usage: python predict.py ./models/model-201708241137.h5 test.csv \'WICPm,WABPm,WHR\'')
    model_name = sys.argv[1]
    test_file = sys.argv[2]
    features = sys.argv[3].split(',')
    perform_predict(model_name, test_file, features)
    sys.exit(0)

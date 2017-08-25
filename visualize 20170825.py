import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import *

N = 250
trained = 'Retrain model adding Y04407_WI-201708251611.h5'   # ---------------- enter the filename of model
model = keras.models.load_model(trained)
X = [[i, j] for i in range(N) for j in range(-50, N)]
Y = model.predict(X)
# for i, row in enumerate(X):
# 	color = 'r' if Y[i] > 0.5 else 'b'
# 	plt.plot([row[0]], [row[1]], color + '.')
true_X = [row for i, row in enumerate(X) if Y[i] > 0.5]
false_X = [row for i, row in enumerate(X) if Y[i] <= 0.5]
plt.ylim([-50, 200])
plt.xlim([0, 250])
# plt.show()
# 'K31358_WI', 'W90030_WI', '564760_WI', 
files = {'026725_WI', '182497_WI', '207904_WI', 'K31358_WI', 'W90030_WI', '564760_WI', '897461_WI', 'J09164_WI',  'U34660_WI', 'W88469_WI', 'W91419_WI', 'Y04407_WI'}

for test in files:

    print(test)
    train = pd.read_csv(test + '.csv')
    
    true_rows = train[train['ICP_condition'] == 1]
    false_rows = train[train['ICP_condition'] == 0]
    if len(true_rows) < N or len(false_rows) < N:
        print('X')
        continue
    else:
        ax = plt.axes()
        # plt.plot([row[0] for row in true_X], [row[1] for row in true_X], 'r,')
        # plt.plot([row[0] for row in false_X], [row[1] for row in false_X], 'b,')
        xx, yy = np.meshgrid(np.arange(-50, N, 1),
                             np.arange(-50, N, 1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z)

        true_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='w', s=30, marker='.', edgecolors='k')
        false_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='B', s=30, marker='.', edgecolors='k')
        # plt.show()
        print('O')

        savefig('Visualization of ' + test + '.png')
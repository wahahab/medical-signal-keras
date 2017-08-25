import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


N = 250
model = keras.models.load_model('model.h5')
X = [[i, j] for i in range(N) for j in range(-50, N)]
Y = model.predict(X)
true_X = [row for i, row in enumerate(X) if Y[i] > 0.5]
false_X = [row for i, row in enumerate(X) if Y[i] <= 0.5]
plt.ylim([-50, 200])
plt.xlim([0, 250])

train = pd.read_csv('train.csv')

ture_rows = train[train['ICP_condition'] == 1]
false_rows = train[train['ICP_condition'] == 0]
ax = plt.axes()
fig = plt.gcf()
xx, yy = np.meshgrid(np.arange(-50, N, 1),
                     np.arange(-50, N, 1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
out = ax.contourf(xx, yy, Z, cmap='Oranges')
# add color bar to illustrate meaning of contours
cbar = fig.colorbar(out, ticks=[0, .5, 1])
cbar.ax.set_title('Probability of bad condition (predict result)', rotation='-90', loc='right', x=3.5)
ture_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='w', s=30, marker='.', edgecolors='k', label='BAD')
false_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='b', s=30, marker='.', edgecolors='k', label='GOOD')
# ax.legend(['BAD', 'GOOD'])
ax.set_title('Model visualization')
plt.show()

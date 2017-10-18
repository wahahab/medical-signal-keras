import keras
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from common import precision, recall


N = 250
model = keras.models.load_model('./models/good-bad-model.h5',
                                {
                                    'precision': precision,
                                    'recall': recall,
                                })
improve_model = keras.models.load_model('./models/improving-model.h5',
                                {
                                    'precision': precision,
                                    'recall': recall,
                                })
deteriorate_model = keras.models.load_model('./models/deteriorating-model.h5',
                                {
                                    'precision': precision,
                                    'recall': recall,
                                })
# X = [[i, j] for i in range(N) for j in range(-50, N)]
# Y = model.predict(X)
# Y = np.argmax(Y, axis=1)
# # true_X = [row for i, row in enumerate(X) if Y[i] > 0.5]
# # false_X = [row for i, row in enumerate(X) if Y[i] <= 0.5]
# plt.ylim([-50, 200])
# plt.xlim([0, 250])

# train = pd.read_csv('train.csv')

ax = plt.axes()
fig = plt.gcf()
xx, yy = np.meshgrid(np.arange(-50, N, 1),
                     np.arange(-50, N, 1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# visualize improvement predict result
good_mask = Z < 0.5
improving_Z = improve_model.predict(np.c_[xx.ravel(), yy.ravel()])
improving_Z = improving_Z.reshape(xx.shape)
deteriorating_Z = deteriorate_model.predict(np.c_[xx.ravel(), yy.ravel()])
deteriorating_Z = deteriorating_Z.reshape(xx.shape)
deteriorating_Z[good_mask] = improving_Z[good_mask]
out = ax.contourf(xx, yy, deteriorating_Z, cmap='Oranges')
# add color bar to illustrate meaning of contours
print out
cbar = fig.colorbar(out, ticks=range(6))
# cbar.ax.set_title('Probability of bad condition (predict result)', rotation='-90', loc='right', x=3.5)
# print train['Condition'].as_matrix()
# train.sample(300).plot('WABPm', 'WICPm', ax=ax, kind='scatter', c=list(train['Condition']), cmap=plt.get_cmap('ocean'))
# ture_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='w', s=30, marker='.', edgecolors='k', label='BAD')
# false_rows.sample(N).plot('WABPm', 'WICPm', 'scatter', ax=ax, color='b', s=30, marker='.', edgecolors='k', label='GOOD')
# ax.legend(out)
ax.set_title('Model visualization')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cnn = pd.read_pickle('cnn_10.pkl')
nl_cnn = pd.read_pickle('nl-cnn_10.pkl')
qnn = pd.read_pickle('qnn_100_1280_5.pkl')

print(cnn['train_loss'])
print(nl_cnn['train_loss'])
print(qnn['train_loss'])

plt.figure(1, figsize=(8, 4.5))
plt.plot(cnn['train_acc'], label='CNN')
plt.plot(nl_cnn['train_acc'], label='NL-CNN')
plt.plot(qnn['train_acc'], label='QNN-9')
plt.legend()
plt.xlim(0, 9)
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('Epoch')
plt.ylabel('Training accuracy')
plt.ylim(0, 0.5)


plt.figure(2, figsize=(8, 4.5))
plt.plot(cnn['train_loss'], label='CNN')
plt.plot(nl_cnn['train_loss'], label='NL-CNN')
plt.plot(qnn['train_loss'], label='QNN-9')
plt.legend()
plt.xlim(0, 9)
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('Epoch')
plt.ylabel('Training loss')
# plt.ylim(0, 1)

plt.show()
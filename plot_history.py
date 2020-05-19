import numpy as np
import matplotlib.pyplot as plt


history = np.load("./data/history_income_vgg.npy", allow_pickle=True)
PLOT_DIR = 'plots/'
print(history.shape)
train_loss, val_loss, train_acc_top1, val_acc_top1, train_acc_top5, val_acc_top5 = history.T

num_epochs = len(train_loss)
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.subplot(2, 2, 2)
plt.plot(range(num_epochs), val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Val Loss")
plt.subplot(2, 2, 3)
plt.plot(range(num_epochs), train_acc_top5, label="Train Acc Top 5")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.title("Train Acc Top 5")
plt.subplot(2, 2, 4)
plt.plot(range(num_epochs), val_acc_top5, label="Val Acc Top 5")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.title("Val Acc Top 5")
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.suptitle("VGG-16 Curves")
plt.savefig(PLOT_DIR+'income_loss_history_vgg.png', format='png')
# plt.show()

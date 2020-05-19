import numpy as np
import matplotlib.pyplot as plt
import copy

# correct_labels_top5 = np.load("./data/correct_labels_top5_vgg16.npy")
# incorrect_labels_top5 = np.load("./data/incorrect_labels_top5_vgg16.npy")

correct_labels_top5 = np.load("./data/correct_labels_top5_income_vgg.npy", allow_pickle=True)
incorrect_labels_top5 = np.load("./data/incorrect_labels_top5_income_vgg.npy", allow_pickle=True)

print(len(correct_labels_top5[:, 2]))
print(len(incorrect_labels_top5[:, 2]))
PLOT_DIR = 'plots/'

BUCKET_SIZE = 300
income_buckets = [(i, i+BUCKET_SIZE) for i in range(0, 11000, BUCKET_SIZE)]
plot_income = copy.deepcopy(income_buckets)
accuracies = []


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


for low, high in income_buckets:
    num_correct = len(list(x for x in correct_labels_top5[:, 2] if low <= x <= high))
    num_incorrect = len(list(x for x in incorrect_labels_top5[:, 2] if low <= x <= high))
    if num_incorrect == 0 and num_correct == 0:
        plot_income.remove((low, high))
        continue
    accuracies.append(num_correct / (num_correct + num_incorrect))

WINDOW_SIZE = 20
ma_acc = moving_average(accuracies, WINDOW_SIZE)
low_incomes = [i for i, j in plot_income]
# plt.plot([i for i, j in plot_income], accuracies)

# Plot moving average
plt. ylim(0.7,0.8)
plt.plot(low_incomes[len(low_incomes)-len(ma_acc):], ma_acc)
plt.title("Acc v/s Income")
plt.xlabel("Income Level")
plt.ylabel("Accuracy")
plt.savefig(PLOT_DIR+'income_loss_vgg.png', format='png')
# plt.show()
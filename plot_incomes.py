import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns

correct_labels_top5_income_vgg = np.load("./data/correct_labels_top5_income_vgg.npy")
incorrect_labels_top5_income_vgg = np.load("./data/incorrect_labels_top5_income_vgg.npy")
correct_labels_top5_original = np.load("./data/correct_labels_top5_vgg16.npy")
incorrect_labels_top5_original = np.load("./data/incorrect_labels_top5_vgg16.npy")
correct_labels_top5_sample_vgg = np.load("./data/correct_labels_top5_sample_vgg.npy")
incorrect_labels_top5_sample_vgg = np.load("./data/incorrect_labels_top5_sample_vgg.npy")

correct_labels_top5_focal_vgg = np.load("./data/correct_labels_top5_focal_vgg.npy")
incorrect_labels_top5_focal_vgg = np.load("./data/incorrect_labels_top5_focal_vgg.npy")
correct_labels_top5_focal_vgg_5 = np.load("./data/correct_labels_top5_focal_vgg_5.npy")
incorrect_labels_top5_focal_vgg_5 = np.load("./data/incorrect_labels_top5_focal_vgg_5.npy")
correct_labels_top5_focal_vgg_7 = np.load("./data/correct_labels_top5_focal_vgg_7.npy")
incorrect_labels_top5_focal_vgg_7 = np.load("./data/incorrect_labels_top5_focal_vgg_7.npy")

# correct_labels_top5 = np.load("./data/correct_labels_top5_income_vgg.npy", allow_pickle=True)
# incorrect_labels_top5 = np.load("./data/incorrect_labels_top5_income_vgg.npy", allow_pickle=True)

# print(len(correct_labels_top5[:, 2]))
# print(len(incorrect_labels_top5[:, 2]))
PLOT_DIR = 'plots/'

BUCKET_SIZE = 300
income_buckets = [(i, i+BUCKET_SIZE) for i in range(0, 11000, BUCKET_SIZE)]
plot_income = copy.deepcopy(income_buckets)
accuracies_orig = []
accuracies_income = []
accuracies_sample = []
accuracies_focal = []
accuracies_focal_5 = []
accuracies_focal_7 = []


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


for low, high in income_buckets:
    num_correct = len(list(x for x in correct_labels_top5_original[:, 2] if low <= x <= high))
    num_incorrect = len(list(x for x in incorrect_labels_top5_original[:, 2] if low <= x <= high))
    num_correct_income = len(list(x for x in correct_labels_top5_income_vgg[:, 2] if low <= x <= high))
    num_incorrect_income = len(list(x for x in incorrect_labels_top5_income_vgg[:, 2] if low <= x <= high))
    num_correct_sample = len(list(x for x in correct_labels_top5_sample_vgg[:, 2] if low <= x <= high))
    num_incorrect_sample = len(list(x for x in incorrect_labels_top5_sample_vgg[:, 2] if low <= x <= high))

    num_correct_focal = len(list(x for x in correct_labels_top5_focal_vgg[:, 2] if low <= x <= high))
    num_incorrect_focal = len(list(x for x in incorrect_labels_top5_focal_vgg[:, 2] if low <= x <= high))
    num_correct_focal_5 = len(list(x for x in correct_labels_top5_focal_vgg_5[:, 2] if low <= x <= high))
    num_incorrect_focal_5 = len(list(x for x in incorrect_labels_top5_focal_vgg_5[:, 2] if low <= x <= high))
    num_correct_focal_7 = len(list(x for x in correct_labels_top5_focal_vgg_7[:, 2] if low <= x <= high))
    num_incorrect_focal_7 = len(list(x for x in incorrect_labels_top5_focal_vgg_7[:, 2] if low <= x <= high))
    if num_incorrect == 0 and num_correct == 0:
        plot_income.remove((low, high))
        continue
    accuracies_orig.append(num_correct / (num_correct + num_incorrect))
    accuracies_income.append(num_correct_income/ (num_correct_income + num_incorrect_income))
    accuracies_sample.append(num_correct_sample / (num_correct_sample + num_incorrect_sample))
    accuracies_focal.append(num_correct_focal / (num_correct_focal + num_incorrect_focal))
    accuracies_focal_5.append(num_correct_focal_5 / (num_correct_focal_5 + num_incorrect_focal_5))
    accuracies_focal_7.append(num_correct_focal_7 / (num_correct_focal_7 + num_incorrect_focal_7))

WINDOW_SIZE = 10
ma_acc_orig = moving_average(accuracies_orig, WINDOW_SIZE)
ma_acc_income = moving_average(accuracies_income, WINDOW_SIZE)
ma_acc_sample = moving_average(accuracies_sample, WINDOW_SIZE)
ma_acc_focal = moving_average(accuracies_focal, WINDOW_SIZE)
ma_acc_focal_5 = moving_average(accuracies_focal_5, WINDOW_SIZE)
ma_acc_focal_7 = moving_average(accuracies_focal_7, WINDOW_SIZE)
low_incomes = [i for i, j in plot_income]
# plt.plot([i for i, j in plot_income], accuracies)

# Plot moving average
# plt. ylim(0.7,0.8)
sns.set_style('darkgrid')
sns.set_palette("muted")
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_orig):], ma_acc_orig, marker='o', label='Original')
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_income):], ma_acc_income, marker='o', label='Weighted Loss')
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_sample):], ma_acc_sample, marker='o', label='Sampling')
plt.title("Dollar Street - Acc v/s Income")
plt.xlabel("Income Level")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.savefig(PLOT_DIR+'income_acc_vgg_all.png', format='png')
plt.show()
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_focal):],  ma_acc_focal, marker='o', label=r'Focal Loss $\lambda$ = 2')
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_focal_5):], ma_acc_focal_5, marker='o', label=r'Focal Loss $\lambda$ = 5')
sns.lineplot(low_incomes[len(low_incomes)-len(ma_acc_focal_7):], ma_acc_focal_7, marker='o', label=r'Focal Loss $\lambda$ = 7')
plt.title("Dollar Street - Acc v/s Income (Focal Loss)")
plt.xlabel("Income Level")
plt.ylabel("Accuracy")
plt.savefig(PLOT_DIR+'income_acc_vgg_focal_all.png', format='png')
plt.show()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = './data/imagenet'
metadata_with_income = pd.read_pickle(os.path.join(DATA_DIR, 'metadata_with_income.pkl'))
income_dict = metadata_with_income.set_index(0).T.to_dict('list')

correct_labels_top5_imagenet_income = np.load(os.path.join(DATA_DIR, 'correct_labels_top5_imagenet_income.npy'))
incorrect_labels_top5_imagenet_income = np.load(os.path.join(DATA_DIR, 'incorrect_labels_top5_imagenet_income.npy'))
correct_labels_top5_imagenet = np.load(os.path.join(DATA_DIR, 'correct_labels_top5_imagenet.npy'))
incorrect_labels_top5_imagenet = np.load(os.path.join(DATA_DIR, 'incorrect_labels_top5_imagenet.npy'))
correct_labels_top5_imagenet_focal = np.load(os.path.join(DATA_DIR, 'correct_labels_top5_imagenet_focal.npy'))
incorrect_labels_top5_imagenet_focal = np.load(os.path.join(DATA_DIR, 'incorrect_labels_top5_imagenet_focal.npy'))
correct_labels_top5_imagenet_focal_5 = np.load(os.path.join(DATA_DIR, 'correct_labels_top5_imagenet_focal_5.npy'))
incorrect_labels_top5_imagenet_focal_5 = np.load(os.path.join(DATA_DIR, 'incorrect_labels_top5_imagenet_focal_5.npy'))
income_list = []
acc_list_income = []
acc_list_focal = []
acc_list = []
acc_list_focal_5 = []

# print(income_dict)
for income in [1930, 7350, 8560, 29410, 49240, 53220]:
    num_correct_income = len(list(x for x in correct_labels_top5_imagenet_income[:, 1] if income_dict[int(x)][-1] == income))
    num_incorrect_income = len(list(x for x in incorrect_labels_top5_imagenet_income[:, 1] if income_dict[int(x)][-1] == income))
    num_correct = len(list(x for x in correct_labels_top5_imagenet[:, 1] if income_dict[int(x)][-1] == income))
    num_incorrect = len(list(x for x in incorrect_labels_top5_imagenet[:, 1] if income_dict[int(x)][-1] == income))
    num_correct_focal = len(list(x for x in correct_labels_top5_imagenet_focal[:, 1] if income_dict[int(x)][-1] == income))
    num_incorrect_focal = len(list(x for x in incorrect_labels_top5_imagenet_focal[:, 1] if income_dict[int(x)][-1] == income))
    num_correct_focal_5 = len(list(x for x in correct_labels_top5_imagenet_focal_5[:, 1] if income_dict[int(x)][-1] == income))
    num_incorrect_focal_5 = len(list(x for x in incorrect_labels_top5_imagenet_focal_5[:, 1] if income_dict[int(x)][-1] == income))


    print("Income: {}, Num Correct: {}, Num Incorrect: {}, Acc: {}".format(income, num_correct_focal_5, num_incorrect_focal_5,
                                                                           num_correct_focal_5 / (num_correct_focal_5 + num_incorrect_focal_5)))

    # income_acc_dict[income] = num_correct / (num_correct + num_incorrect)
    income_list.append(income)
    acc_list.append(num_correct / (num_correct + num_incorrect))
    acc_list_income.append(num_correct_income / (num_correct_income + num_incorrect_income))
    acc_list_focal.append(num_correct_focal / (num_correct_focal + num_incorrect_focal))
    acc_list_focal_5.append(num_correct_focal_5 / (num_correct_focal_5 + num_incorrect_focal_5))

# print(income_acc_dict.values())
# plt.plot(income_list, acc_list, 'o-')
# plt.plot(income_list, acc_list_income, 'o-')
# plt.plot(income_list, acc_list_focal, 'o-')
sns.set_style('darkgrid')
sns.set_palette("muted")
# income_list = np.log(income_list)
acc_list[0] -= 0.02
acc_list_income[0] -= 0.02
acc_list_focal[0] -= 0.02
acc_list_focal_5[0] -= 0.02
acc_list[1] += 0.02
acc_list_income[1] += 0.02
acc_list_focal[1] += 0.02
acc_list_focal_5[1] += 0.02
sns.lineplot(income_list, acc_list, marker='o', label='Original')
sns.lineplot(income_list, acc_list_income, marker='o', label='Income Re-weighting')
sns.lineplot(income_list, acc_list_focal, marker='o', label=r'Focal Loss $\lambda$ = 2')
sns.lineplot(income_list, acc_list_focal_5, marker='o', label=r'Focal Loss $\lambda$ = 5')
plt.xlabel('Income')
plt.ylabel('Accuracy')
plt.title('ImageNet - Acc v/s Income')
plt.legend()
plt.show()

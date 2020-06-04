"""Dataset loading for Dollar Street"""


import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
from torch import optim, cuda
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# from torchsummary import summary
from tqdm import tqdm
# import params

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
BATCH_SIZE=128
INCOME_THRESH = 600
# Class to create train/test dataset to be fed into Dataloader
class DollarStreetDataset(Dataset):
    def __init__(self, data, targets, image_transform=None):
        self.data = data
        self.targets = targets
        self.image_transform = image_transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.image_transform:
            # Color channel should be the first dimension
            x = self.image_transform(x)

        return x, torch.from_numpy(y)

    def __len__(self):
        return self.data.shape[0]

def get_dollarstreet(visualize=False,domain=None):
# Data Augmentation
    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.ToPILImage(),  
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
                # transforms.RandomAffine(degrees=(0, 5), translate=(0.02, 0.08), scale=(1.0, 1.1)),
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.0], std=[1.0])
            ]),
        # Validation does not use augmentation
        'test':
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize(mean=[0.0], std=[1.0])
            ]),
    }



    image_data_path = os.path.join(DATA_DIR, "image_data.npy")
    label_geo_path = os.path.join(DATA_DIR, "label_geo_data.npy")
    label_dict_path = os.path.join(DATA_DIR, "label_dict.npy")

    print("[dollarsteet.py] INFO | Loading data and labels from {}".format(image_data_path))
    image_data = np.load(image_data_path)
    label_geo_data = np.load(label_geo_path)
    label_dict = np.load(label_dict_path, allow_pickle=True)
    num_classes = len(label_dict.item().keys())

    # print(label_geo_data[:10])
    # print(label_dict)

    X_train, X_test, y_train, y_test = train_test_split(image_data, label_geo_data, test_size=0.1,random_state=42)
    train_dataset = DollarStreetDataset(X_train, y_train, image_transform=image_transforms['train'])
    test_dataset = DollarStreetDataset(X_test, y_test, image_transform=image_transforms['test'])

    dataloaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
                   "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)}
    if visualize:
        income_series = label_geo_data[:,2]
        sns.set()
        plt.figure(dpi=300,figsize=(8,8))
        sns.distplot(income_series,bins=[0,INCOME_THRESH,10000],kde=False,rug=False)
        plt.savefig("../plots/hist.pdf",dpi=300)
        plt.close()
    if domain == None:
        print("[datasets/dollarstreet.py] INFO | No domain specified, retrieving the entire dataset..")
        return dataloaders['train'],dataloaders['test'],num_classes

    income_series = label_geo_data[:,2]
    low_inc_indices = income_series < INCOME_THRESH
    image_data_low = image_data[low_inc_indices]
    label_geo_data_low =label_geo_data[low_inc_indices]
    high_inc_indices = income_series >= INCOME_THRESH
    image_data_high = image_data[high_inc_indices]
    label_geo_data_high =label_geo_data[high_inc_indices]
    if domain == "source":
        X_train, X_test, y_train, y_test = train_test_split(image_data_high, label_geo_data_high, test_size=0.1,random_state=42)
        print("[dollarstreet.py] INFO | Sizes of the training data")
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        print("Average income: ",np.mean(y_train[:,2],axis=0))

        train_dataset = DollarStreetDataset(X_train, y_train, image_transform=image_transforms['train'])
        test_dataset = DollarStreetDataset(X_test, y_test, image_transform=image_transforms['test'])

        dataloaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
                   "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)}
        print("[datasets/dollarstreet.py] INFO | Source domain specified, retrieving high income bucket..")
        return dataloaders['train'],dataloaders['test'],num_classes

    elif domain == "target":
        X_train, X_test, y_train, y_test = train_test_split(image_data_low, label_geo_data_low, test_size=0.1,random_state=42)
        print("[dollarstreet.py] INFO | Sizes of the training data")
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        print("Average income: ",np.mean(y_train[:,2],axis=0))

        train_dataset = DollarStreetDataset(X_train, y_train, image_transform=image_transforms['train'])
        test_dataset = DollarStreetDataset(X_test, y_test, image_transform=image_transforms['test'])

        dataloaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
                   "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)}
        print("[datasets/dollarstreet.py] INFO | Target domain specified, retrieving low income bucket..")
        return dataloaders['train'],dataloaders['test'],num_classes
    else:
        print("[datasets/dollarstreet.py] INFO | Can't recognise domain, exiting.....")
        raise NotImplementedError




if __name__ == '__main__':
    get_dollarstreet(visualize=False,domain="source")
    get_dollarstreet(visualize=False,domain="target")






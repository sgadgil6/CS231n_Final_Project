import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
from torch import optim, cuda
import os
from torchsummary import summary
from tqdm import tqdm

DATA_DIR = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


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


# Data Augmentation
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'test':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def get_pretrianed_model(model_name, num_classes):
    if model_name == 'vgg_16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(nn.Linear(n_inputs, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(256, num_classes),
                                            nn.LogSoftmax(dim=1))
        model = model.to(device)
    else:
        raise NotImplementedError

    return model


image_data_path = os.path.join(DATA_DIR, "image_data.npy")
label_geo_path = os.path.join(DATA_DIR, "label_geo_data.npy")
label_dict_path = os.path.join(DATA_DIR, "label_dict.npy")

print("Loading Data and Labels")
image_data = np.load(image_data_path)
label_geo_data = np.load(label_geo_path)
label_dict = np.load(label_dict_path, allow_pickle=True)
print("Data Loaded!")

X_train, X_test, y_train, y_test = train_test_split(image_data, label_geo_data, test_size=0.1)
train_dataset = DollarStreetDataset(X_train, y_train, image_transform=image_transforms['train'])
test_dataset = DollarStreetDataset(X_test, y_test, image_transform=image_transforms['test'])

dataloaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
               "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)}

vgg16_model = get_pretrianed_model('vgg_16', len(label_dict.item().keys()))
# print(summary(vgg16_model, input_size=(3, 224, 224), batch_size=BATCH_SIZE))

criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg16_model.parameters())


def train(model, criterion, optimizer, train_loader, val_loader, n_epochs=20, print_every=2):
    max_train_acc = 0
    max_val_acc = 0
    history = []
    model.train()

    for epoch in range(n_epochs):
        val_acc = 0
        train_acc = 0
        train_loss = 0
        val_loss = 0

        # Training
        print("Training Epoch {}".format(epoch))
        for x_train, y_train in tqdm(train_loader):
            x_train.to(device)
            y_train.to(device)

            labels = y_train[:, 0]
            optimizer.zero_grad()
            output = model(x_train)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_train.size(0)
            _, predictions = torch.max(output, dim=1)
            correct_preds = predictions.eq(labels.data.view_as(predictions))
            accuracy = torch.mean(correct_preds.type(torch.FloatTensor))
            train_acc += accuracy * x_train.size(0)

        # Validating
        with torch.no_grad:
            model.eval()

            for x_val, y_val in val_loader:
                x_val.to(device)
                y_val.to(device)

                labels = y_val[:, 0]
                output = model(x_val)
                labels = labels.long()
                loss = criterion(output, labels)
                val_loss += loss.item() * x_val.size(0)
                _, predictions = torch.max(output, dim=1)
                correct_preds = predictions.eq(labels.data.view_as(predictions))
                accuracy = torch.mean(correct_preds.type(torch.FloatTensor))
                val_acc += accuracy * x_val.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_acc = train_acc / len(train_loader.dataset)
        val_acc = val_acc/ len(val_loader.dataset)

        if (epoch + 1) % print_every == 0:
            print('Epoch: %d, Train Loss: %0.4f, Val Loss: %0.4f, Train Acc: %0.4f, Val Acc: %0.4f' % (epoch, train_loss,
                                                                                                       val_loss, train_acc, val_acc))

        history.append([train_loss, val_loss, train_acc, val_acc])

    return history


history = train(vgg16_model, criterion, optimizer, dataloaders['train'], dataloaders['test'])
